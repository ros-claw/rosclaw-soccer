"""Bounded offline receiver actor/critic study from audited strict SIM outcomes."""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from rosclaw_soccer.providers.g1.velocity_policy import VelocityPolicyParameters
from rosclaw_soccer.rsi.anchored_receiving_motor import AnchoredReceivingMotor
from rosclaw_soccer.rsi.receiver_actor_critic import ReceiverCritic, episode_balanced_value_loss
from rosclaw_soccer.rsi.receiver_bounded_update import bounded_actor_update
from rosclaw_soccer.rsi.receiver_strict_terminal_reward import admit_strict_training
from rosclaw_soccer.rsi.velocity_prior_actor import VelocityPriorActor
from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_diagnose_pass_contact_timing import digest
from scripts.rsi_native_fresh_common import clean


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("dataset", "audit", "anchor-admission", "anchor-dataset", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    for key in (
        "dataset-hash",
        "audit-hash",
        "audit-head",
        "anchor-admission-hash",
        "anchor-dataset-hash",
        "expected-head",
    ):
        p.add_argument("--" + key, required=True)
    p.add_argument("--seed", type=int, choices=(92401, 92402, 92403), required=True)
    a = p.parse_args()
    clean(a.expected_head)
    dp = a.dataset / "complete.json"
    anchors = a.anchor_admission.parent / "healthy-anchors.npz"
    inputs = (
        (dp, a.dataset_hash),
        (a.audit, a.audit_hash),
        (a.anchor_admission, a.anchor_admission_hash),
        (a.anchor_dataset, a.anchor_dataset_hash),
    )
    if (
        a.output.exists()
        or a.output.resolve().is_relative_to(Path.cwd())
        or any(path.is_symlink() or digest(path) != h for path, h in inputs)
    ):
        raise ValueError("exact audited receipts and unused external output required")
    data, audit, admission, old_data = [json.loads(path.read_text()) for path, _ in inputs]
    sufficiency = admit_strict_training(data["records"])
    if (
        data.get("status") != "RECEIVER_ONPOLICY_PHYSICAL_EPISODES_NOT_TRAINED"
        or data.get("reward_profile") != "center-strict-full-history-v1"
        or any(
            data.get(k) is not False
            for k in ("optimizer_authorized", "promotion_authorized", "hardware_authorized")
        )
        or data.get("research_data_sufficiency") != sufficiency
        or not sufficiency["research_training_data_sufficient"]
        or audit.get("status") != "RECEIVER_ONPOLICY_READBACK_AND_DENSITY_NOT_TRAINED"
        or audit.get("dataset_hash") != a.dataset_hash
        or audit.get("source_head") != a.audit_head
        or audit.get("reward_profile") != "center-strict-full-history-v1"
        or audit.get("research_data_sufficiency") != sufficiency
        or audit.get("exact_inputs_actions_rewards_and_source_replayed") is not True
        or audit.get("episodes") != 72
        or any(
            audit.get(k) is not False
            for k in ("optimizer_authorized", "promotion_authorized", "hardware_authorized")
        )
        or admission.get("status") != "RECEIVER_RESEARCH_TRAINING_DATA_ADMITTED_NOT_TRAINED"
        or any(
            admission.get(k) is not False for k in ("promotion_authorized", "hardware_authorized")
        )
        or admission.get("dataset_hash") != a.anchor_dataset_hash
        or admission.get("healthy_anchor_controls") != 5201
        or old_data.get("model_identity") != data.get("model_identity")
        or old_data.get("status") != "RECEIVER_ONPOLICY_PHYSICAL_EPISODES_NOT_TRAINED"
        or anchors.is_symlink()
        or digest(anchors) != admission.get("healthy_anchor_hash")
    ):
        raise ValueError(
            "strict positive/negative data, independent audit and matching old anchors required"
        )
    torch.manual_seed(a.seed)
    torch.set_num_threads(2)
    model, mh, config, ch = data["model_identity"]
    params = VelocityPolicyParameters.load(Path(config), expected_hash=ch)
    prior = VelocityPriorActor.from_onnx(Path(model), expected_hash=mh, reference_features=14)
    scale = torch.tensor(params.action_scale, dtype=torch.float32)
    actor = AnchoredReceivingMotor(prior, scale).eval()
    critic = ReceiverCritic()
    with np.load(anchors, allow_pickle=False) as f:
        anchor_x, anchor_r = (
            torch.from_numpy(f["inputs"].copy()),
            torch.from_numpy(f["reference"].copy()),
        )
    if len(anchor_x) != 5201:
        raise ValueError("all original audited healthy anchors required")
    episodes, paths = [], []
    for row in data["records"]:
        path = a.dataset / row["path"]
        if (
            path.is_symlink()
            or path.resolve().parent != a.dataset.resolve()
            or digest(path) != row["hash"]
        ):
            raise ValueError("exact original sampled episode required")
        paths.append((path, row["hash"]))
        with np.load(path, allow_pickle=False) as e:
            n = row["controls"]
            if (
                n < 1
                or len(e["inputs"]) != n
                or not np.array_equal(e["action_scale"], scale.numpy())
                or e["rewards"][-1] != row["terminal_reward"]
                or np.any(e["rewards"][:-1] != 0)
            ):
                raise ValueError("actual sampled history and strict terminal reward required")

            def t(name: str) -> torch.Tensor:
                return torch.from_numpy(e[name].copy())

            returns = torch.tensor(
                [row["terminal_reward"] * 0.99 ** (n - 1 - i) for i in range(n)],
                dtype=torch.float32,
            )
            episodes.append(
                dict(
                    observations=t("inputs"),
                    actions=t("sampled_action"),
                    old_means=t("old_mean"),
                    reference=t("reference"),
                    old_logp=t("old_logp"),
                    action_scale=scale,
                    returns=returns,
                    sampling_law="box-truncated-ar1",
                    all_actor_actions_actually_executed=True,
                    density_audited=True,
                    full_task_outcome_observed=True,
                    assisted=False,
                    external_teacher=False,
                    authority_rejected=False,
                )
            )
    if (
        len(episodes) != 72
        or sum(len(e["observations"]) for e in episodes) != data["sampled_controls"]
    ):
        raise ValueError("complete audited episode history required")
    observations = torch.cat([e["observations"] for e in episodes])
    targets = torch.cat([e["returns"] for e in episodes])
    optimizer = torch.optim.Adam(critic.parameters(), lr=0.001)
    losses = []
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        loss = episode_balanced_value_loss(critic, episodes)
        if not torch.isfinite(loss):
            raise ValueError("nonfinite critic objective")
        loss.backward()
        if any(p.grad is None or not torch.isfinite(p.grad).all() for p in critic.parameters()):
            raise ValueError("nonfinite critic gradient")
        optimizer.step()
        losses.append(float(loss.detach()))
    critic.eval().requires_grad_(False)
    with torch.no_grad():
        advantages = targets - critic(observations)
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-6)
    start = 0
    for ep in episodes:
        end = start + len(ep["observations"])
        ep["advantages"] = advantages[start:end]
        start = end
    result = bounded_actor_update(
        actor,
        critic,
        episodes,
        torch.full((29,), math.log(0.02)),
        anchor_x,
        anchor_r,
        step_size=1e-4,
    )
    if any(
        digest(path) != h
        for path, h in (
            *inputs,
            *paths,
            (anchors, admission["healthy_anchor_hash"]),
            (Path(model), mh),
            (Path(config), ch),
        )
    ):
        raise ValueError("research inputs changed during optimization")
    clean(a.expected_head)
    a.output.mkdir(parents=True)
    checkpoint = a.output / "research-candidate.pt"
    torch.save(
        dict(
            actor=actor.plastic.state_dict(),
            critic=critic.state_dict(),
            seed=a.seed,
            dataset_hash=a.dataset_hash,
            audit_hash=a.audit_hash,
            physical_retention_proven=False,
            promotion_authorized=False,
            hardware_authorized=False,
        ),
        checkpoint,
    )
    write_once(
        a.output / "complete.json",
        dict(
            status="CENTER_RECEIVER_OFFLINE_CANDIDATE"
            if result["accepted"]
            else "CENTER_RECEIVER_OFFLINE_REJECTED",
            source_head=a.expected_head,
            dataset_hash=a.dataset_hash,
            audit_hash=a.audit_hash,
            anchor_admission_hash=a.anchor_admission_hash,
            checkpoint_hash=digest(checkpoint),
            seed=a.seed,
            sampled_controls=data["sampled_controls"],
            critic_updates=3,
            critic_losses=losses,
            declared_initial_actor_step=1e-4,
            **result,
            actual_new_physics=False,
            online_rl=False,
            goal_complete=False,
        ),
    )


if __name__ == "__main__":
    main()
