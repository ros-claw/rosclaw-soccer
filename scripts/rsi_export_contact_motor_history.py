"""Bind causal locomotor history to audited continuous receiving examples."""

import argparse
import json
from pathlib import Path

import numpy as np

from rosclaw_soccer.rsi.receiving_causal_history import augment_foot_with_velocity_history
from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_diagnose_pass_contact_timing import digest
from scripts.rsi_native_fresh_common import clean
from scripts.rsi_review_relay_balance_counterfactual import bind, report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pair", nargs=2, action="append", required=True, metavar=("ROOT", "AUDIT"))
    for name in ("dataset", "continuous", "continuous-audit", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    for name in ("dataset-hash", "continuous-hash", "continuous-audit-hash", "expected-head"):
        p.add_argument("--" + name, required=True)
    a = p.parse_args()
    clean(a.expected_head)
    cp, dp = a.continuous / "complete.json", a.continuous / "continuous.npz"
    checks = (
        (a.dataset, a.dataset_hash),
        (cp, a.continuous_hash),
        (a.continuous_audit, a.continuous_audit_hash),
    )
    if (
        len(a.pair) != 6
        or a.output.exists()
        or a.output.resolve().is_relative_to(Path.cwd())
        or any(path.is_symlink() or digest(path) != expected for path, expected in checks)
        or dp.is_symlink()
    ):
        raise ValueError("exact source data and unused external output required")
    dataset, receipt, audit = (json.loads(path.read_text()) for path, _ in checks)
    if (
        receipt.get("status") != "RELAY_CONTINUOUS_CONTACT_DATA_NOT_TRAINED"
        or audit.get("status") != "RELAY_CONTINUOUS_CONTACT_FINITE_DIFFERENCE_AUDIT_NOT_CONTROL"
        or receipt.get("dataset_hash") != a.dataset_hash
        or audit.get("features_hash") != a.continuous_hash
        or audit.get("all_labels_independently_recomputed") is not True
        or audit.get("all_features_independently_finite_differenced") is not True
        or digest(dp) != receipt["data_hash"]
    ):
        raise ValueError("fully audited original continuous data required")
    with np.load(dp, allow_pickle=False) as f:
        foot, groups, ids, times = (
            f[k].copy() for k in ("features", "context", "condition", "control")
        )
    n = receipt["samples"]
    if (
        foot.shape != (n, 12)
        or any(v.shape != (n,) for v in (groups, ids, times))
        or not np.isfinite(foot).all()
    ):
        raise ValueError("complete finite audited original inputs required")
    originals = {(r["context"], r["side"]): (i, r) for i, r in enumerate(receipt["records"])}
    pins = {(r["block"], r["expanded"]): r for r in dataset["source_pins"]}
    if len(originals) != 72 or len(pins) != 6:
        raise ValueError("all distinct source conditions required")
    features = np.zeros((n, 308), dtype=np.float32)
    used, seen, blocks, frozen, rows = set(), set(), set(), [], []
    for root_name, audit_name in a.pair:
        root, own = Path(root_name), Path(audit_name)
        decpath = root / "declaration.json"
        dec = json.loads(decpath.read_text())
        expanded = dec["protocol"] == "soccer.neural_relay_balance_expanded.development.v6"
        if not expanded and dec["protocol"] != "soccer.neural_relay_balance_teacher.development.v5":
            raise ValueError("original known protocol required")
        key = (dec["development_context_block"], expanded)
        if key in blocks or key not in pins:
            raise ValueError("unique audited source blocks required")
        blocks.add(key)
        status = (
            "NEURAL_RELAY_BALANCE_EXPANDED64_COLLECTED_PENDING_OWN_REPLAY"
            if expanded
            else "NEURAL_RELAY_BALANCE_DEVELOPMENT32_COLLECTED_PENDING_OWN_REPLAY"
        )
        source, _, sh = report(
            root, own, pins[key]["audit_hash"], status, worlds=64 if expanded else 32
        )
        if sh != pins[key]["source_hash"] or digest(decpath) != source["declaration_hash"]:
            raise ValueError("original independently replayed producer required")
        for trial in source["records"]:
            if trial["course"]["arm"]["balance_teacher"] != -1:
                continue
            condition = (trial["course"]["context"], trial["course"]["lateral_m"])
            if condition not in originals or condition in seen:
                raise ValueError("unique original condition required")
            seen.add(condition)
            index, row = originals[condition]
            start, count = row["first_sample"], row["sample_count"]
            if (
                row["block"] != key[0]
                or row["expanded"] != expanded
                or row["source_trial"] != trial["index"]
                or start < 0
                or count < 1
                or start + count > n
                or not np.array_equal(ids[start : start + count], np.full(count, index))
                or not np.array_equal(groups[start : start + count], np.full(count, condition[0]))
                or not np.array_equal(times[start : start + count], np.arange(count))
            ):
                raise ValueError("exact causal sample boundaries and source binding required")
            velocity = bind(root, trial, "receiver_velocity")
            if velocity is None:
                raise ValueError("own measured motor observation history required")
            with np.load(velocity, allow_pickle=False) as f:
                obs, control = f["obs"], f["control"]
                if (
                    obs.ndim != 2
                    or obs.shape[1] != 98
                    or obs.dtype != np.float32
                    or len(obs) < count
                    or not np.isfinite(obs).all()
                    or not np.array_equal(control, np.arange(len(obs)))
                ):
                    raise ValueError("exact contiguous own 98D motor input stream required")
                features[start : start + count] = augment_foot_with_velocity_history(
                    foot[start : start + count], obs[:count]
                )
            if digest(velocity) != trial["artifacts"]["receiver_velocity"]["hash"]:
                raise ValueError("motor history changed during export")
            used.update(range(start, start + count))
            rows.append(
                dict(
                    context=condition[0],
                    side=condition[1],
                    block=key[0],
                    expanded=expanded,
                    source_trial=trial["index"],
                    velocity_hash=digest(velocity),
                    first_sample=start,
                    sample_count=count,
                )
            )
        frozen.extend(((root / "complete.json", sh), (own, pins[key]["audit_hash"])))
    if (
        blocks != set(pins)
        or seen != set(originals)
        or used != set(range(n))
        or any(digest(path) != expected for path, expected in checks)
        or digest(dp) != receipt["data_hash"]
        or any(digest(path) != h for path, h in frozen)
    ):
        raise ValueError("complete unchanged causal provenance required")
    a.output.mkdir(parents=True)
    path = a.output / "motor-history.npz"
    np.savez_compressed(path, features=features)
    clean(a.expected_head)
    write_once(
        a.output / "complete.json",
        dict(
            status="RELAY_CONTACT_HISTORY_DATA_NOT_TRAINED",
            source_head=a.expected_head,
            continuous_hash=a.continuous_hash,
            continuous_audit_hash=a.continuous_audit_hash,
            feature_hash=digest(path),
            conditions=72,
            samples=n,
            features=308,
            records=rows,
            future_observations_in_features=False,
            future_labels_in_features=False,
            independent_physics_reintegration=False,
            promotion_authorized=False,
            hardware_authorized=False,
        ),
    )


if __name__ == "__main__":
    main()
