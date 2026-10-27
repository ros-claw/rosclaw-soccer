"""Grouped offline value fitting on audited paired SIM relay outcomes."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from rosclaw_soccer.rsi.relay_balance_selector import decision_rewards, select_with_parent_fallback
from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_diagnose_pass_contact_timing import digest
from scripts.rsi_native_fresh_common import clean


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("dataset", "audit", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("dataset-hash", "audit-hash", "expected-head"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    clean(args.expected_head)
    rp, ap, dp = (
        args.dataset / "complete.json",
        args.audit / "complete.json",
        args.dataset / "dataset.npz",
    )
    if (
        args.output.exists()
        or args.output.resolve().is_relative_to(Path.cwd())
        or any(p.is_symlink() for p in (rp, ap, dp))
        or digest(rp) != args.dataset_hash
        or digest(ap) != args.audit_hash
    ):
        raise ValueError("exact audited decision dataset and unused output required")
    receipt, audit = (json.loads(p.read_text()) for p in (rp, ap))
    if (
        receipt.get("status") != "PAIRED_RELAY_BALANCE_DECISIONS_EXPORTED_NOT_TRAINED"
        or audit.get("status") != "PAIRED_RELAY_BALANCE_DECISIONS_READBACK_NOT_REINTEGRATED"
        or audit.get("dataset_hash") != args.dataset_hash
        or digest(dp) != receipt["dataset_hash"]
    ):
        raise ValueError("complete paired data and readback required")
    with np.load(dp, allow_pickle=False) as f:
        x, groups = f["features"].copy(), f["context"].copy()
        successes, failures, kicks = f["success"].copy(), f["failure"].copy(), f["kick"].copy()
    if (
        x.shape != (len(groups), 17)
        or any(
            a.shape != (len(groups), 4) or a.dtype != np.bool_ for a in (successes, failures, kicks)
        )
        or not np.isfinite(x).all()
        or len(groups) != receipt["handoff_decisions"]
        or len(set(groups.tolist())) < 2
    ):
        raise ValueError("finite causal paired features and measured outcome labels required")
    rewards = np.stack(
        [decision_rewards(successes[i], failures[i], kicks[i]) for i in range(len(groups))]
    )
    contexts = sorted(int(c) for c in set(groups))
    if any(int((groups != context).sum()) < 8 for context in contexts):
        raise ValueError("insufficient independent training decisions; no fitting")
    args.output.mkdir(parents=True)
    write_once(
        args.output / "declaration.json",
        dict(
            dataset_hash=args.dataset_hash,
            audit_hash=args.audit_hash,
            source_head=args.expected_head,
            contexts=contexts,
            seeds=[17, 43, 71],
            split="leave one original passer context out including all sides and four options",
            inputs="17 causal body-relative state and communicated goal",
            reward="2*precise_healthy + .25*foot_strike - 2*either_body_failure",
            training="17-32-tanh-4 outcome regression, 500 full-batch Adam updates per model",
            fallback="parent when nearest-training RMS distance > 2 or predicted gain < .25",
            no_handoff_conditions=receipt["no_handoff_conditions"],
            offline=True,
            controller_authorized=False,
            hardware_authorized=False,
        ),
    )
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    evaluations = []
    for context in contexts:
        train, test = groups != context, groups == context
        center, scale = x[train].mean(axis=0), np.maximum(x[train].std(axis=0), 0.05)
        z = np.clip((x - center) / scale, -8, 8).astype(np.float32)
        support = z[train].copy()
        xt, yt = torch.from_numpy(z[train]), torch.from_numpy(rewards[train])
        for seed in (17, 43, 71):
            torch.manual_seed(seed)
            model = torch.nn.Sequential(
                torch.nn.Linear(17, 32), torch.nn.Tanh(), torch.nn.Linear(32, 4)
            )
            optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
            for _ in range(500):
                loss = torch.nn.functional.mse_loss(model(xt), yt)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            with torch.no_grad():
                values = model(torch.from_numpy(z[test])).numpy()
            if not np.isfinite(values).all() or not bool(torch.isfinite(loss)):
                raise ValueError("finite trained model outputs required")
            checkpoint = args.output / f"context-{context}-seed-{seed}.npz"
            np.savez_compressed(
                checkpoint,
                center=center,
                scale=scale,
                support=support,
                weight1=model[0].weight.detach().numpy(),
                bias1=model[0].bias.detach().numpy(),
                weight2=model[2].weight.detach().numpy(),
                bias2=model[2].bias.detach().numpy(),
            )
            indices = np.flatnonzero(test)
            choices = []
            for local, index in enumerate(indices):
                distance = float(np.sqrt(np.mean((support - z[index]) ** 2, axis=1)).min())
                action, reason = select_with_parent_fallback(values[local], distance)
                choices.append(
                    dict(
                        sample=int(index),
                        action=action,
                        reason=reason,
                        support_distance=distance,
                        parent_success=bool(successes[index, 0]),
                        chosen_success=bool(successes[index, action]),
                        parent_failure=bool(failures[index, 0]),
                        chosen_failure=bool(failures[index, action]),
                        parent_kick=bool(kicks[index, 0]),
                        chosen_kick=bool(kicks[index, action]),
                        outcomes_are_existing_counterfactuals_not_new_physics=True,
                    )
                )
            row = dict(
                context=context,
                seed=seed,
                training_decisions=int(train.sum()),
                heldout_decisions=int(test.sum()),
                checkpoint=checkpoint.name,
                checkpoint_hash=digest(checkpoint),
                choices=choices,
            )
            evaluations.append(row)
            write_once(args.output / f"context-{context}-seed-{seed}.json", row)
    if (
        digest(rp) != args.dataset_hash
        or digest(ap) != args.audit_hash
        or digest(dp) != receipt["dataset_hash"]
    ):
        raise ValueError("decision data changed during training")
    clean(args.expected_head)
    write_once(
        args.output / "complete.json",
        dict(
            status="RELAY_BALANCE_OFFLINE_SELECTOR_HOLDOUT_NOT_PHYSICS",
            source_head=args.expected_head,
            dataset_hash=args.dataset_hash,
            audit_hash=args.audit_hash,
            contexts=contexts,
            rows=evaluations,
            optimizer_updates=len(evaluations) * 500,
            offline=True,
            online_rl=False,
            physics_evaluated_as_controller=False,
            safety_certificate=False,
            hardware_authorized=False,
            promotion_authorized=False,
        ),
    )


if __name__ == "__main__":
    main()
