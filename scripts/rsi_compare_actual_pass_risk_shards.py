"""Compare audited sixteen-context risk models on the original held-out data."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from rosclaw_soccer.rsi.actual_pass_contact_risk import contact_risk_data
from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_compare_actual_pass_risk_models import counts, predict
from scripts.rsi_native_context_recovery_common import digest
from scripts.rsi_native_fresh_common import clean


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("baseline", "baseline-audit", "data", "expanded", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    for name in (
        "baseline-hash",
        "baseline-audit-hash",
        "data-hash",
        "expanded-hash",
        "expected-head",
    ):
        parser.add_argument(f"--{name}", required=True)
    for name in ("shard", "shard-audit"):
        parser.add_argument(f"--{name}", type=Path, action="append", required=True)
    for name in ("shard-hash", "shard-audit-hash"):
        parser.add_argument(f"--{name}", action="append", required=True)
    args = parser.parse_args()
    clean(args.expected_head)
    if any(
        len(v) != 4 for v in (args.shard, args.shard_audit, args.shard_hash, args.shard_audit_hash)
    ):
        raise ValueError("all four trained and independently replayed shards required")
    roots = (
        args.baseline,
        args.baseline_audit,
        args.data,
        args.expanded,
        *args.shard,
        *args.shard_audit,
    )
    hashes = (
        args.baseline_hash,
        args.baseline_audit_hash,
        args.data_hash,
        args.expanded_hash,
        *args.shard_hash,
        *args.shard_audit_hash,
    )
    paths = [root / "complete.json" for root in roots]
    if (
        args.output.exists()
        or args.output.resolve().is_relative_to(Path.cwd())
        or any(p.is_symlink() or digest(p) != h for p, h in zip(paths, hashes, strict=True))
    ):
        raise ValueError("exact model, audit, data receipts and unused output required")
    receipts = [json.loads(p.read_text()) for p in paths]
    baseline, baseline_audit, original, expanded = receipts[:4]
    shards, audits = receipts[4:8], receipts[8:12]
    if (
        baseline["status"] != "ACTUAL_PASS_CONTACT_RISK_SUPERVISED_CONTEXT_HOLDOUT_NOT_CONTROL"
        or baseline["dataset_hash"] != args.data_hash
        or baseline_audit["status"]
        != "ACTUAL_PASS_CONTACT_RISK_HOLDOUT_PREDICTIONS_REPLAYED_NOT_CERTIFIED"
        or baseline_audit["source_hash"] != args.baseline_hash
    ):
        raise ValueError("audited original risk model required")
    by_key = {}
    full_holdouts = []
    for i, (model, audit) in enumerate(zip(shards, audits, strict=True)):
        if (
            model["status"] != "ACTUAL_PASS_CONTACT_RISK_SUPERVISED_CONTEXT_HOLDOUT_NOT_CONTROL"
            or model["dataset_hash"] != args.expanded_hash
            or model["shard_count"] != 4
            or model["shard_index"] != i
            or audit["status"]
            != "ACTUAL_PASS_CONTACT_RISK_HOLDOUT_PREDICTIONS_REPLAYED_NOT_CERTIFIED"
            or audit["source_hash"] != args.shard_hash[i]
            or audit["dataset_hash"] != args.expanded_hash
        ):
            raise ValueError("all four exact context shards and prediction replays required")
        for row in model["rows"]:
            key = (row["heldout_context"], row["seed"])
            if key in by_key:
                raise ValueError("no duplicated held-out models")
            by_key[key] = (args.shard[i], row)
            full_holdouts.append(
                dict(
                    context=key[0],
                    seed=key[1],
                    tp=row["tp"],
                    fn=row["fn"],
                    fp=row["fp"],
                    tn=row["tn"],
                )
            )
    if len(by_key) != 48 or len({key[0] for key in by_key}) != 16:
        raise ValueError("all sixteen contexts and three seeds required")
    dp, ep = args.data / "dataset.npz", args.expanded / "dataset.npz"
    if (
        dp.is_symlink()
        or ep.is_symlink()
        or digest(dp) != original["dataset_hash"]
        or digest(ep) != expanded["dataset_hash"]
    ):
        raise ValueError("exact numeric datasets required")
    with np.load(dp, allow_pickle=False) as f:
        x, y, groups = contact_risk_data({k: f[k].copy() for k in f.files}, original["courses"])
    with np.load(ep, allow_pickle=False) as f:
        all_data = {k: f[k].copy() for k in f.files}
    xx, yy, gg = contact_risk_data(all_data, expanded["courses"])
    subset = all_data["source_index"] == 2
    if (
        not np.array_equal(x, xx[subset])
        or not np.array_equal(y, yy[subset])
        or not np.array_equal(groups, gg[subset])
    ):
        raise ValueError("same exact original held-out physical transitions required")
    old_rows = {(r["heldout_context"], r["seed"]): r for r in baseline["rows"]}
    expected = {(c, s) for c in (1, 4, 20, 21) for s in (17, 43, 71)}
    if set(old_rows) != expected or not expected.issubset(by_key):
        raise ValueError("all original held-out contexts and seeds required")
    torch.set_num_threads(1)
    results = []
    for context, seed in sorted(expected):
        mask = groups == context
        truth = y[mask].astype(bool)
        root, row = by_key[(context, seed)]
        old = counts(predict(args.baseline, old_rows[(context, seed)], x[mask]), truth)
        new = counts(predict(root, row, x[mask]), truth)
        if any(old_rows[(context, seed)][k] != v for k, v in old.items()):
            raise ValueError("baseline counts must reproduce original held-out evaluation")
        results.append(
            dict(
                context=context,
                seed=seed,
                samples=int(mask.sum()),
                positives=int(truth.sum()),
                baseline=old,
                expanded=new,
            )
        )
    if (
        any(digest(p) != h for p, h in zip(paths, hashes, strict=True))
        or digest(dp) != original["dataset_hash"]
        or digest(ep) != expanded["dataset_hash"]
    ):
        raise ValueError("input changed during matched evaluation")
    args.output.mkdir()
    write_once(
        args.output / "complete.json",
        dict(
            status="ACTUAL_PASS_RISK_16_CONTEXT_MATCHED_COMPARISON_NOT_CONTROL",
            source_head=args.expected_head,
            baseline_hash=args.baseline_hash,
            baseline_audit_hash=args.baseline_audit_hash,
            data_hash=args.data_hash,
            expanded_hash=args.expanded_hash,
            shard_hashes=args.shard_hash,
            shard_audit_hashes=args.shard_audit_hash,
            matched_original=results,
            all_context_holdouts=full_holdouts,
            exact_same_heldout_transitions=True,
            correlated_transitions_not_independent_games=True,
            safety_certificate=False,
            controller_interface=False,
            promotion_authorized=False,
            hardware_authorized=False,
        ),
    )


if __name__ == "__main__":
    main()
