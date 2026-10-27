"""One bounded, offline SIM-only actor/critic feasibility update, not promotion."""

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
    for name in ("dataset", "admission", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    for name in ("dataset-hash", "admission-hash", "expected-head"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--device", choices=("cpu", "cuda:0"), default="cpu")
    p.add_argument("--actor-step", type=float, choices=(1e-7, 1e-4), default=1e-7)
    a = p.parse_args()
    clean(a.expected_head)
    dp, ap = a.dataset / "complete.json", a.admission / "complete.json"
    if (
        a.output.exists()
        or a.output.resolve().is_relative_to(Path.cwd())
        or any(x.is_symlink() for x in (dp, ap))
        or digest(dp) != a.dataset_hash
        or digest(ap) != a.admission_hash
    ):
        raise ValueError("exact admitted receipts and unused external output required")
    data, admission = json.loads(dp.read_text()), json.loads(ap.read_text())
    anchor_path = a.admission / "healthy-anchors.npz"
    if (
        admission.get("status") != "RECEIVER_RESEARCH_TRAINING_DATA_ADMITTED_NOT_TRAINED"
        or admission.get("offline_research_training_authorized") is not True
        or admission.get("dataset_hash") != a.dataset_hash
        or admission.get("episodes") != 72
        or admission.get("conditions") != 24
        or admission.get("sampled_controls") != 18482
        or admission.get("healthy_anchor_controls") != 5201
        or admission.get("promotion_authorized") is not False
        or admission.get("hardware_authorized") is not False
        or anchor_path.is_symlink()
        or digest(anchor_path) != admission.get("healthy_anchor_hash")
    ):
        raise ValueError("exact independent research admission and anchors required")
    if a.device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("requested GPU unavailable")
    torch.manual_seed(92301)
    torch.set_num_threads(2)
    device = torch.device(a.device)
    model, model_hash, config, config_hash = data["model_identity"]
    params = VelocityPolicyParameters.load(Path(config), expected_hash=config_hash)
    prior = VelocityPriorActor.from_onnx(
        Path(model), expected_hash=model_hash, reference_features=14
    )
    scale = torch.tensor(params.action_scale, dtype=torch.float32)
    actor = AnchoredReceivingMotor(prior, scale).to(device).eval()
    critic = ReceiverCritic().to(device)
    with np.load(anchor_path, allow_pickle=False) as f:
        anchor_x = torch.from_numpy(f["inputs"].copy()).to(device)
        anchor_r = torch.from_numpy(f["reference"].copy()).to(device)
    rows = data["records"]
    if len(rows) != 72 or len({(r["context"], r["side"], r["seed"]) for r in rows}) != 72:
        raise ValueError("all unique audited sampled episodes required")
    episodes, total_controls = [], 0
    for row in rows:
        path = a.dataset / row["path"]
        if (
            path.is_symlink()
            or path.resolve().parent != a.dataset.resolve()
            or digest(path) != row["hash"]
        ):
            raise ValueError("confined exact audited episode required")
        with np.load(path, allow_pickle=False) as e:
            n = row["controls"]
            if n != len(e["inputs"]) or not np.array_equal(e["action_scale"], scale.numpy()):
                raise ValueError("unchanged own history and action scale required")
            if not np.isclose(e["rewards"][-1], row["terminal_reward"]) or np.any(
                e["rewards"][:-1] != 0
            ):
                raise ValueError("exact terminal-only reward required")

            def t(name: str) -> torch.Tensor:
                return torch.from_numpy(e[name].copy()).to(device)

            returns = torch.tensor(
                [row["terminal_reward"] * 0.99 ** (n - 1 - i) for i in range(n)],
                dtype=torch.float32,
                device=device,
            )
            episodes.append(
                dict(
                    observations=t("inputs"),
                    actions=t("sampled_action"),
                    old_means=t("old_mean"),
                    reference=t("reference"),
                    old_logp=t("old_logp"),
                    action_scale=scale.to(device),
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
            total_controls += n
    if total_controls != 18482 or len(anchor_x) != 5201:
        raise ValueError("full admitted history and healthy retention controls required")
    observations = torch.cat([e["observations"] for e in episodes])
    targets = torch.cat([e["returns"] for e in episodes])
    optimizer = torch.optim.Adam(critic.parameters(), lr=0.001)
    critic_losses = []
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        loss = episode_balanced_value_loss(critic, episodes)
        if not torch.isfinite(loss):
            raise ValueError("nonfinite critic objective")
        loss.backward()
        if any(p.grad is None or not torch.isfinite(p.grad).all() for p in critic.parameters()):
            raise ValueError("nonfinite critic gradient")
        optimizer.step()
        critic_losses.append(float(loss.detach()))
    critic.eval().requires_grad_(False)
    with torch.no_grad():
        advantages = targets - critic(observations)
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-6)
    start = 0
    for ep in episodes:
        end = start + len(ep["observations"])
        ep["advantages"] = advantages[start:end]
        start = end
    log_std = torch.full((29,), math.log(0.02), dtype=torch.float32, device=device)
    result = bounded_actor_update(
        actor, critic, episodes, log_std, anchor_x, anchor_r, step_size=a.actor_step
    )
    if any(
        digest(x) != h
        for x, h in (
            (dp, a.dataset_hash),
            (ap, a.admission_hash),
            (anchor_path, admission["healthy_anchor_hash"]),
            (Path(model), model_hash),
            (Path(config), config_hash),
        )
    ):
        raise ValueError("evidence changed during offline computation")
    if any(digest(a.dataset / row["path"]) != row["hash"] for row in rows):
        raise ValueError("episode changed during offline computation")
    clean(a.expected_head)
    a.output.mkdir(parents=True)
    checkpoint = a.output / "research-candidate.pt"
    torch.save(
        dict(
            actor=actor.plastic.state_dict(),
            critic=critic.state_dict(),
            seed=92301,
            dataset_hash=a.dataset_hash,
            admission_hash=a.admission_hash,
            physical_retention_proven=False,
            promotion_authorized=False,
            hardware_authorized=False,
        ),
        checkpoint,
    )
    write_once(
        a.output / "complete.json",
        dict(
            status="RECEIVER_OFFLINE_RESEARCH_CANDIDATE"
            if result["accepted"]
            else "RECEIVER_OFFLINE_ACTOR_REJECTED",
            source_head=a.expected_head,
            dataset_hash=a.dataset_hash,
            admission_hash=a.admission_hash,
            checkpoint_hash=digest(checkpoint),
            sampled_controls=total_controls,
            critic_updates=3,
            critic_losses=critic_losses,
            declared_initial_actor_step=a.actor_step,
            **result,
            actual_new_physics=False,
            online_rl=False,
        ),
    )


if __name__ == "__main__":
    main()
