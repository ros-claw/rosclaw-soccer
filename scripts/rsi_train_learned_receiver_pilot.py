"""Continue a SIM receiver from its own replayed pilot; no promotion or deployment."""

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
from rosclaw_soccer.rsi.velocity_prior_actor import VelocityPriorActor
from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_diagnose_pass_contact_timing import digest
from scripts.rsi_native_fresh_common import clean


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    for name in (
        "dataset",
        "dataset-audit",
        "parent",
        "parent-audit",
        "parent-dataset",
        "anchor-admission",
        "anchor-dataset",
        "output",
    ):
        p.add_argument("--" + name, type=Path, required=True)
    for name in (
        "dataset-hash",
        "dataset-audit-hash",
        "parent-hash",
        "parent-audit-hash",
        "parent-dataset-hash",
        "anchor-admission-hash",
        "anchor-dataset-hash",
        "expected-head",
    ):
        p.add_argument("--" + name, required=True)
    a = p.parse_args()
    clean(a.expected_head)
    pins = (
        (a.dataset / "complete.json", a.dataset_hash),
        (a.dataset_audit, a.dataset_audit_hash),
        (a.parent, a.parent_hash),
        (a.parent_audit, a.parent_audit_hash),
        (a.parent_dataset, a.parent_dataset_hash),
        (a.anchor_admission, a.anchor_admission_hash),
        (a.anchor_dataset, a.anchor_dataset_hash),
    )
    if (
        a.output.exists()
        or a.output.resolve().is_relative_to(Path.cwd())
        or any(p.is_symlink() or digest(p) != h for p, h in pins)
    ):
        raise ValueError("exact pinned simulation inputs and unused external output required")
    data, audit, parent_audit, parent_data, admission, anchor_data = [
        json.loads(p.read_text())
        for p in (pins[0][0], pins[1][0], pins[3][0], pins[4][0], pins[5][0], pins[6][0])
    ]
    anchors = a.anchor_admission.parent / "healthy-anchors.npz"
    if (
        data.get("status") != "LEARNED_RECEIVER_PILOT_PHYSICAL_DATA_NOT_TRAINED"
        or data.get("receiver_checkpoint_hash") != a.parent_hash
        or data.get("episodes") != 24
        or data.get("conditions") != 8
        or audit.get("status") != "LEARNED_RECEIVER_PILOT_DATA_READBACK_AUDITED"
        or audit.get("dataset_hash") != a.dataset_hash
        or audit.get("episodes") != 24
        or audit.get("controls") != data.get("sampled_controls")
        or parent_audit.get("status") != "DENSE_RECEIVER_CANDIDATE_READBACK_MATCH"
        or parent_audit.get("checkpoint_hash") != a.parent_hash
        or parent_audit.get("dataset_hash") != a.parent_dataset_hash
        or parent_data.get("model_identity") != data.get("model_identity")
        or admission.get("status") != "RECEIVER_RESEARCH_TRAINING_DATA_ADMITTED_NOT_TRAINED"
        or admission.get("dataset_hash") != a.anchor_dataset_hash
        or anchor_data.get("model_identity") != data.get("model_identity")
        or admission.get("healthy_anchor_controls") != 5201
        or anchors.is_symlink()
        or digest(anchors) != admission.get("healthy_anchor_hash")
        or any(
            x.get(k) is not False
            for x in (data, audit, parent_audit, admission)
            for k in ("promotion_authorized", "hardware_authorized")
        )
    ):
        raise ValueError("own audited pilot, exact behavior parent and retention anchors required")
    state = torch.load(a.parent, map_location="cpu", weights_only=True)
    if state.get("dataset_hash") != a.parent_dataset_hash or any(
        state.get(k) is not False
        for k in ("physical_retention_proven", "promotion_authorized", "hardware_authorized")
    ):
        raise ValueError("exact unpromoted behavior checkpoint required")
    torch.set_num_threads(2)
    model, mh, config, ch = data["model_identity"]
    params = VelocityPolicyParameters.load(Path(config), expected_hash=ch)
    scale = torch.tensor(params.action_scale, dtype=torch.float32)
    actor = AnchoredReceivingMotor(
        VelocityPriorActor.from_onnx(Path(model), expected_hash=mh, reference_features=14), scale
    ).eval()
    critic = ReceiverCritic()
    actor.plastic.load_state_dict(state["actor"], strict=True)
    critic.load_state_dict(state["critic"], strict=True)
    if any(not torch.isfinite(p).all() for p in (*actor.parameters(), *critic.parameters())):
        raise ValueError("finite exact behavior actor and retained critic required")
    with np.load(anchors, allow_pickle=False) as f:
        anchor_x, anchor_r = (
            torch.from_numpy(f["inputs"].copy()),
            torch.from_numpy(f["reference"].copy()),
        )
    episodes, frozen, keys = [], [], set()
    for row in data["records"]:
        key = (row["context"], row["side"], row["seed"])
        path = a.dataset / row["path"]
        if (
            key in keys
            or path.is_symlink()
            or path.resolve().parent != a.dataset.resolve()
            or digest(path) != row["hash"]
        ):
            raise ValueError("unique confined independently audited episodes required")
        keys.add(key)
        frozen.append((path, row["hash"]))
        with np.load(path, allow_pickle=False) as ep:
            n = row["controls"]
            if (
                len(ep["inputs"]) != n
                or ep["dense_rewards"].shape != (n,)
                or not np.array_equal(ep["action_scale"], scale.numpy())
            ):
                raise ValueError("exact actual sampled motor data and aligned rewards required")
            reward = ep["rewards"].astype(np.float32) + np.float32(0.2) * ep["dense_rewards"]
            returns, running = np.empty(n, np.float32), 0.0
            for i in range(n - 1, -1, -1):
                running = float(reward[i]) + 0.99 * running
                returns[i] = running
            episodes.append(
                dict(
                    observations=torch.from_numpy(ep["inputs"].copy()),
                    actions=torch.from_numpy(ep["sampled_action"].copy()),
                    old_means=torch.from_numpy(ep["old_mean"].copy()),
                    reference=torch.from_numpy(ep["reference"].copy()),
                    old_logp=torch.from_numpy(ep["old_logp"].copy()),
                    action_scale=scale,
                    returns=torch.from_numpy(returns),
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
        len(episodes) != 24
        or sum(len(e["observations"]) for e in episodes) != data["sampled_controls"]
    ):
        raise ValueError("complete reviewed pilot history required")
    optimizer, losses = torch.optim.Adam(critic.parameters(), lr=0.001), []
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        loss = episode_balanced_value_loss(critic, episodes)
        loss.backward()
        if not torch.isfinite(loss) or any(
            p.grad is None or not torch.isfinite(p.grad).all() for p in critic.parameters()
        ):
            raise ValueError("finite critic loss and gradient required")
        optimizer.step()
        losses.append(float(loss.detach()))
    critic.eval().requires_grad_(False)
    with torch.no_grad():
        observations = torch.cat([e["observations"] for e in episodes])
        targets = torch.cat([e["returns"] for e in episodes])
        adv = targets - critic(observations)
        adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-6)
    start = 0
    for ep in episodes:
        end = start + len(ep["observations"])
        ep["advantages"] = adv[start:end]
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
        digest(p) != h for p, h in (*pins, *frozen, (anchors, admission["healthy_anchor_hash"]))
    ):
        raise ValueError("all inputs must remain immutable during update")
    clean(a.expected_head)
    a.output.mkdir(parents=True)
    checkpoint = a.output / "research-candidate.pt"
    torch.save(
        dict(
            actor=actor.plastic.state_dict(),
            critic=critic.state_dict(),
            parent_hash=a.parent_hash,
            dataset_hash=a.dataset_hash,
            dataset_audit_hash=a.dataset_audit_hash,
            physical_retention_proven=False,
            promotion_authorized=False,
            hardware_authorized=False,
        ),
        checkpoint,
    )
    write_once(
        a.output / "complete.json",
        dict(
            status="LEARNED_RECEIVER_PILOT_CONTINUATION_CANDIDATE"
            if result["accepted"]
            else "LEARNED_RECEIVER_PILOT_CONTINUATION_REJECTED",
            source_head=a.expected_head,
            parent_hash=a.parent_hash,
            dataset_hash=a.dataset_hash,
            dataset_audit_hash=a.dataset_audit_hash,
            checkpoint_hash=digest(checkpoint),
            episodes=24,
            conditions=8,
            sampled_controls=data["sampled_controls"],
            critic_updates=3,
            critic_losses=losses,
            dense_weight=0.2,
            gamma=0.99,
            **result,
            fresh_holdout=False,
            actual_new_physics=False,
            online_rl=False,
            goal_complete=False,
        ),
    )


if __name__ == "__main__":
    main()
