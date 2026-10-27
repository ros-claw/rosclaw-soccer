"""Offline held-out-context 20ms contact prediction; never controls a motor."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from rosclaw_soccer.rsi.actual_pass_contact_risk import contact_risk_data
from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_native_context_recovery_common import digest
from scripts.rsi_native_fresh_common import clean


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("dataset", "audit", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    for name in ("dataset-hash", "audit-hash", "expected-head"):
        parser.add_argument(f"--{name}", required=True)
    args = parser.parse_args()
    clean(args.expected_head)
    receipt_path, audit_path = args.dataset / "complete.json", args.audit / "complete.json"
    if (
        args.output.exists()
        or args.output.resolve().is_relative_to(Path.cwd())
        or receipt_path.is_symlink()
        or audit_path.is_symlink()
        or digest(receipt_path) != args.dataset_hash
        or digest(audit_path) != args.audit_hash
    ):
        raise ValueError("exact data and unused external training output required")
    receipt, audit = (json.loads(p.read_text()) for p in (receipt_path, audit_path))
    path = args.dataset / "dataset.npz"
    if (
        receipt["status"] != "ACTUAL_PASS_CAUSAL_TRANSITIONS_EXPORTED_NOT_TRAINED"
        or audit["status"] != "ACTUAL_PASS_CAUSAL_TRANSITIONS_READBACK_NOT_REINTEGRATED"
        or audit["dataset_hash"] != args.dataset_hash
        or path.is_symlink()
        or digest(path) != receipt["dataset_hash"]
        or receipt["qualified_episodes"] != 0
    ):
        raise ValueError(
            "exact source-readback dataset, explicit zero successful episodes required"
        )
    with np.load(path, allow_pickle=False) as f:
        x, y, groups = contact_risk_data({k: f[k].copy() for k in f.files}, receipt["courses"])
    if len(x) != receipt["transitions"] or not 0 < int(y.sum()) < len(y):
        raise ValueError("exact declared grouped contact dataset required")
    folds = sorted(int(value) for value in set(groups))
    args.output.mkdir()
    write_once(
        args.output / "declaration.json",
        dict(
            dataset_hash=args.dataset_hash,
            audit_hash=args.audit_hash,
            source_head=args.expected_head,
            folds=folds,
            seeds=[17, 43, 71],
            training="supervised binary classifier, 282-64-tanh-1, 200 full-batch Adam steps",
            label="receiver recorded non-foot contact >1N within next 10 native substeps",
            inputs=(
                "before-action 79 qpos, 76 qvel, 98 velocity observations "
                "and 29 actual applied normalized actions"
            ),
            split="leave one shared starting context out, all paired arms held together",
            scale="training context mean/std only, floor 0.01, clip 10",
            threshold=0.5,
            positive_weight="negative/positive capped at 20, training context only",
            no_successful_demonstrations=True,
            safety_certificate=False,
            controller_interface=False,
            hardware_authorized=False,
        ),
    )
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    rows = []
    for holdout in folds:
        train, test = groups != holdout, groups == holdout
        center = x[train].mean(axis=0)
        scale = np.maximum(x[train].std(axis=0), 0.01)
        z = np.clip((x - center) / scale, -10, 10).astype(np.float32)
        xt, yt = torch.from_numpy(z[train]), torch.from_numpy(y[train, None])
        xv, truth = torch.from_numpy(z[test]), y[test].astype(bool)
        positives, negatives = int(yt.sum()), int(len(yt) - yt.sum())
        if not positives or not negatives or not len(truth):
            raise ValueError("both training classes and held-out context required")
        weight = torch.tensor([min(20.0, negatives / positives)])
        for seed in (17, 43, 71):
            torch.manual_seed(seed)
            network = torch.nn.Sequential(
                torch.nn.Linear(282, 64), torch.nn.Tanh(), torch.nn.Linear(64, 1)
            )
            optimizer = torch.optim.Adam(network.parameters(), lr=0.003)
            for _ in range(200):
                loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    network(xt), yt, pos_weight=weight
                )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            with torch.no_grad():
                probability = torch.sigmoid(network(xv)).numpy().reshape(-1)
            prediction = probability >= 0.5
            checkpoint = args.output / f"context-{holdout}-seed-{seed}.npz"
            np.savez_compressed(
                checkpoint,
                center=center,
                scale=scale,
                weight1=network[0].weight.detach().numpy(),
                bias1=network[0].bias.detach().numpy(),
                weight2=network[2].weight.detach().numpy(),
                bias2=network[2].bias.detach().numpy(),
            )
            row = dict(
                heldout_context=holdout,
                seed=seed,
                train_rows=int(train.sum()),
                heldout_rows=int(test.sum()),
                train_positive=positives,
                heldout_positive=int(truth.sum()),
                tp=int((prediction & truth).sum()),
                fn=int((~prediction & truth).sum()),
                fp=int((prediction & ~truth).sum()),
                tn=int((~prediction & ~truth).sum()),
                checkpoint=checkpoint.name,
                checkpoint_hash=digest(checkpoint),
            )
            rows.append(row)
            write_once(args.output / f"context-{holdout}-seed-{seed}.json", row)
    if (
        digest(receipt_path) != args.dataset_hash
        or digest(audit_path) != args.audit_hash
        or digest(path) != receipt["dataset_hash"]
    ):
        raise ValueError("training source changed")
    write_once(
        args.output / "complete.json",
        dict(
            status="ACTUAL_PASS_CONTACT_RISK_SUPERVISED_CONTEXT_HOLDOUT_NOT_CONTROL",
            source_head=args.expected_head,
            dataset_hash=args.dataset_hash,
            audit_hash=args.audit_hash,
            rows=rows,
            offline=True,
            online_rl=False,
            physics_evaluated_as_controller=False,
            safety_certificate=False,
            promotion_authorized=False,
            hardware_authorized=False,
        ),
    )


if __name__ == "__main__":
    main()
