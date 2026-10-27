"""Compare two risk models on exactly the same held-out pass transitions."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from rosclaw_soccer.rsi.actual_pass_contact_risk import contact_risk_data
from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_native_context_recovery_common import digest
from scripts.rsi_native_fresh_common import clean


def counts(pred: np.ndarray, truth: np.ndarray) -> dict[str, int]:
    return dict(
        tp=int((pred & truth).sum()),
        fn=int((~pred & truth).sum()),
        fp=int((pred & ~truth).sum()),
        tn=int((~pred & ~truth).sum()),
    )


def predict(root: Path, row: dict, x: np.ndarray) -> np.ndarray:
    path = root / row["checkpoint"]
    if path.is_symlink() or path.parent != root or digest(path) != row["checkpoint_hash"]:
        raise ValueError("exact confined model checkpoint required")
    with np.load(path, allow_pickle=False) as f:
        w = {k: f[k].copy() for k in f.files}
    expected = {
        "center": (282,),
        "scale": (282,),
        "weight1": (64, 282),
        "bias1": (64,),
        "weight2": (1, 64),
        "bias2": (1,),
    }
    if (
        set(w) != set(expected)
        or any(w[k].shape != s or not np.isfinite(w[k]).all() for k, s in expected.items())
        or np.any(w["scale"] < 0.01)
    ):
        raise ValueError("complete finite model checkpoint required")
    z = np.clip((x - w["center"]) / w["scale"], -10, 10).astype(np.float32)
    network = torch.nn.Sequential(torch.nn.Linear(282, 64), torch.nn.Tanh(), torch.nn.Linear(64, 1))
    with torch.no_grad():
        network[0].weight.copy_(torch.from_numpy(w["weight1"]))
        network[0].bias.copy_(torch.from_numpy(w["bias1"]))
        network[2].weight.copy_(torch.from_numpy(w["weight2"]))
        network[2].bias.copy_(torch.from_numpy(w["bias2"]))
        return torch.sigmoid(network(torch.from_numpy(z))).numpy().reshape(-1) >= 0.5


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "baseline",
        "baseline-audit",
        "candidate",
        "candidate-audit",
        "data",
        "aggregate",
        "output",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    for name in (
        "baseline-hash",
        "baseline-audit-hash",
        "candidate-hash",
        "candidate-audit-hash",
        "data-hash",
        "aggregate-hash",
        "expected-head",
    ):
        parser.add_argument(f"--{name}", required=True)
    args = parser.parse_args()
    clean(args.expected_head)
    roots = (
        args.baseline,
        args.baseline_audit,
        args.candidate,
        args.candidate_audit,
        args.data,
        args.aggregate,
    )
    hashes = (
        args.baseline_hash,
        args.baseline_audit_hash,
        args.candidate_hash,
        args.candidate_audit_hash,
        args.data_hash,
        args.aggregate_hash,
    )
    files = [root / "complete.json" for root in roots]
    if (
        args.output.exists()
        or args.output.resolve().is_relative_to(Path.cwd())
        or any(p.is_symlink() or digest(p) != h for p, h in zip(files, hashes, strict=True))
    ):
        raise ValueError("exact models, audits, datasets and unused external output required")
    baseline, ba, candidate, ca, data_receipt, aggregate_receipt = (
        json.loads(p.read_text()) for p in files
    )
    if (
        any(
            r["status"] != "ACTUAL_PASS_CONTACT_RISK_SUPERVISED_CONTEXT_HOLDOUT_NOT_CONTROL"
            for r in (baseline, candidate)
        )
        or any(
            a["status"] != "ACTUAL_PASS_CONTACT_RISK_HOLDOUT_PREDICTIONS_REPLAYED_NOT_CERTIFIED"
            for a in (ba, ca)
        )
        or ba["source_hash"] != args.baseline_hash
        or ca["source_hash"] != args.candidate_hash
        or baseline["dataset_hash"] != args.data_hash
        or candidate["dataset_hash"] != args.aggregate_hash
    ):
        raise ValueError("two audited and correctly bound risk models required")
    data_path, agg_path = args.data / "dataset.npz", args.aggregate / "dataset.npz"
    if (
        data_path.is_symlink()
        or agg_path.is_symlink()
        or digest(data_path) != data_receipt["dataset_hash"]
        or digest(agg_path) != aggregate_receipt["dataset_hash"]
    ):
        raise ValueError("exact numeric datasets required")
    with np.load(data_path, allow_pickle=False) as f:
        x, y, groups = contact_risk_data({k: f[k].copy() for k in f.files}, data_receipt["courses"])
    with np.load(agg_path, allow_pickle=False) as f:
        aggregate = {k: f[k].copy() for k in f.files}
    ax, ay, ag = contact_risk_data(aggregate, aggregate_receipt["courses"])
    subset = aggregate["source_index"] == 2
    if (
        not np.array_equal(x, ax[subset])
        or not np.array_equal(y, ay[subset])
        or not np.array_equal(groups, ag[subset])
    ):
        raise ValueError("candidate and baseline must share the exact original test transitions")
    old_rows = {(r["heldout_context"], r["seed"]): r for r in baseline["rows"]}
    new_rows = {(r["heldout_context"], r["seed"]): r for r in candidate["rows"]}
    keys = {(c, s) for c in (1, 4, 20, 21) for s in (17, 43, 71)}
    if set(old_rows) != keys or set(new_rows) != keys:
        raise ValueError("all paired context/seed models required")
    torch.set_num_threads(1)
    results = []
    for context, seed in sorted(keys):
        mask = groups == context
        truth = y[mask].astype(bool)
        old = predict(args.baseline, old_rows[(context, seed)], x[mask])
        new = predict(args.candidate, new_rows[(context, seed)], x[mask])
        old_counts, new_counts = counts(old, truth), counts(new, truth)
        if any(old_rows[(context, seed)][k] != v for k, v in old_counts.items()):
            raise ValueError("baseline held-out test differs from original receipt")
        results.append(
            dict(
                context=context,
                seed=seed,
                samples=int(mask.sum()),
                positives=int(truth.sum()),
                baseline=old_counts,
                candidate=new_counts,
            )
        )
    if (
        any(digest(p) != h for p, h in zip(files, hashes, strict=True))
        or digest(data_path) != data_receipt["dataset_hash"]
        or digest(agg_path) != aggregate_receipt["dataset_hash"]
    ):
        raise ValueError("input changed during paired evaluation")
    args.output.mkdir()
    write_once(
        args.output / "complete.json",
        dict(
            status="ACTUAL_PASS_RISK_SAME_HELDOUT_TRANSITIONS_COMPARED_NOT_CONTROL",
            source_head=args.expected_head,
            baseline_hash=args.baseline_hash,
            candidate_hash=args.candidate_hash,
            baseline_audit_hash=args.baseline_audit_hash,
            candidate_audit_hash=args.candidate_audit_hash,
            data_hash=args.data_hash,
            aggregate_hash=args.aggregate_hash,
            rows=results,
            exact_same_heldout_transitions=True,
            shared_contexts_not_independent_games=True,
            confidence_calibrated=False,
            controller_interface=False,
            safety_certificate=False,
            promotion_authorized=False,
            hardware_authorized=False,
        ),
    )


if __name__ == "__main__":
    main()
