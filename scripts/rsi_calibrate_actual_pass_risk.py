"""Calibrate contact-warning thresholds using training contexts only.

No model weights, controller, or safety authority are changed. The 0.95 target
is on training positives; held-out performance and false alarms are reported.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from rosclaw_soccer.rsi.actual_pass_contact_risk import contact_risk_data, training_recall_threshold
from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_compare_actual_pass_risk_models import counts, probabilities
from scripts.rsi_native_context_recovery_common import digest
from scripts.rsi_native_fresh_common import clean


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("candidate", "candidate-audit", "data", "aggregate", "comparison", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    for name in (
        "candidate-hash",
        "candidate-audit-hash",
        "data-hash",
        "aggregate-hash",
        "comparison-hash",
        "expected-head",
    ):
        parser.add_argument(f"--{name}", required=True)
    args = parser.parse_args()
    clean(args.expected_head)
    roots = (args.candidate, args.candidate_audit, args.data, args.aggregate, args.comparison)
    hashes = (
        args.candidate_hash,
        args.candidate_audit_hash,
        args.data_hash,
        args.aggregate_hash,
        args.comparison_hash,
    )
    files = [root / "complete.json" for root in roots]
    if (
        args.output.exists()
        or args.output.resolve().is_relative_to(Path.cwd())
        or any(p.is_symlink() or digest(p) != h for p, h in zip(files, hashes, strict=True))
    ):
        raise ValueError("exact model, audit, paired data and unused output required")
    model, audit, original, aggregate, comparison = (json.loads(p.read_text()) for p in files)
    if (
        model["status"] != "ACTUAL_PASS_CONTACT_RISK_SUPERVISED_CONTEXT_HOLDOUT_NOT_CONTROL"
        or model["dataset_hash"] != args.aggregate_hash
        or audit["status"] != "ACTUAL_PASS_CONTACT_RISK_HOLDOUT_PREDICTIONS_REPLAYED_NOT_CERTIFIED"
        or audit["source_hash"] != args.candidate_hash
        or comparison["status"] != "ACTUAL_PASS_RISK_SAME_HELDOUT_TRANSITIONS_COMPARED_NOT_CONTROL"
        or comparison["candidate_hash"] != args.candidate_hash
        or comparison["data_hash"] != args.data_hash
        or comparison["aggregate_hash"] != args.aggregate_hash
    ):
        raise ValueError("audited candidate and exact paired comparison required")
    data_path, agg_path = args.data / "dataset.npz", args.aggregate / "dataset.npz"
    if (
        data_path.is_symlink()
        or agg_path.is_symlink()
        or digest(data_path) != original["dataset_hash"]
        or digest(agg_path) != aggregate["dataset_hash"]
    ):
        raise ValueError("exact numeric source datasets required")
    with np.load(data_path, allow_pickle=False) as f:
        x, y, groups = contact_risk_data({k: f[k].copy() for k in f.files}, original["courses"])
    with np.load(agg_path, allow_pickle=False) as f:
        data = {k: f[k].copy() for k in f.files}
    ax, ay, ag = contact_risk_data(data, aggregate["courses"])
    subset = data["source_index"] == 2
    if (
        not np.array_equal(x, ax[subset])
        or not np.array_equal(y, ay[subset])
        or not np.array_equal(groups, ag[subset])
    ):
        raise ValueError("exact original held-out transitions required")
    models = {(r["heldout_context"], r["seed"]): r for r in model["rows"]}
    matched = {(r["context"], r["seed"]): r for r in comparison["rows"]}
    keys = {(c, s) for c in (1, 4, 20, 21) for s in (17, 43, 71)}
    if set(models) != keys or set(matched) != keys:
        raise ValueError("all context and seed checkpoints required")
    torch.set_num_threads(1)
    rows = []
    for context, seed in sorted(keys):
        train, test = ag != context, groups == context
        train_p = probabilities(args.candidate, models[(context, seed)], ax[train])
        train_y = ay[train].astype(bool)
        threshold = training_recall_threshold(train_p, train_y)
        test_p = probabilities(args.candidate, models[(context, seed)], x[test])
        truth = y[test].astype(bool)
        fixed, calibrated = counts(test_p >= 0.5, truth), counts(test_p >= threshold, truth)
        if fixed != matched[(context, seed)]["candidate"]:
            raise ValueError("same held-out fixed-threshold baseline required")
        rows.append(
            dict(
                context=context,
                seed=seed,
                training_positive=int(train_y.sum()),
                training_threshold=threshold,
                training_positive_detected=int((train_p[train_y] >= threshold).sum()),
                heldout_samples=int(test.sum()),
                heldout_positive=int(truth.sum()),
                original_model=matched[(context, seed)]["baseline"],
                expanded_fixed=fixed,
                expanded_calibrated=calibrated,
            )
        )
    if (
        any(digest(p) != h for p, h in zip(files, hashes, strict=True))
        or digest(data_path) != original["dataset_hash"]
        or digest(agg_path) != aggregate["dataset_hash"]
    ):
        raise ValueError("input changed during calibration")
    args.output.mkdir()
    write_once(
        args.output / "complete.json",
        dict(
            status="ACTUAL_PASS_RISK_TRAIN_ONLY_THRESHOLD_EVALUATED_NOT_CONTROL",
            source_head=args.expected_head,
            candidate_hash=args.candidate_hash,
            comparison_hash=args.comparison_hash,
            data_hash=args.data_hash,
            aggregate_hash=args.aggregate_hash,
            rows=rows,
            train_recall_target=0.95,
            threshold_capped_at=0.5,
            holdout_labels_used_for_threshold=False,
            model_weights_changed=False,
            same_heldout_transitions=True,
            confidence_calibrated=False,
            safety_certificate=False,
            controller_interface=False,
            promotion_authorized=False,
            hardware_authorized=False,
        ),
    )


if __name__ == "__main__":
    main()
