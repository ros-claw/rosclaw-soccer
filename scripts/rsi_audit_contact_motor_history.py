"""Independently verify every causal 98D motor-history feature alignment."""

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_diagnose_pass_contact_timing import digest
from scripts.rsi_native_fresh_common import clean
from scripts.rsi_review_relay_balance_counterfactual import bind, report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pair", nargs=2, action="append", required=True, metavar=("ROOT", "AUDIT"))
    for name in ("dataset", "continuous", "features", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    for name in (
        "dataset-hash",
        "continuous-hash",
        "features-hash",
        "producer-head",
        "expected-head",
    ):
        p.add_argument("--" + name, required=True)
    a = p.parse_args()
    clean(a.expected_head)
    cp, dp = a.continuous / "complete.json", a.continuous / "continuous.npz"
    fp, hp = a.features / "complete.json", a.features / "motor-history.npz"
    checks = ((a.dataset, a.dataset_hash), (cp, a.continuous_hash), (fp, a.features_hash))
    if (
        len(a.pair) != 6
        or a.output.exists()
        or a.output.resolve().is_relative_to(Path.cwd())
        or any(path.is_symlink() or digest(path) != expected for path, expected in checks)
        or dp.is_symlink()
        or hp.is_symlink()
    ):
        raise ValueError("exact source data and unused external output required")
    dataset, continuous, receipt = (json.loads(path.read_text()) for path, _ in checks)
    if (
        receipt.get("status") != "RELAY_CONTACT_HISTORY_DATA_NOT_TRAINED"
        or receipt.get("source_head") != a.producer_head
        or receipt.get("continuous_hash") != a.continuous_hash
        or receipt.get("future_observations_in_features") is not False
        or receipt.get("future_labels_in_features") is not False
        or digest(dp) != continuous["data_hash"]
        or digest(hp) != receipt["feature_hash"]
    ):
        raise ValueError("exact causally declared history and original features required")
    with np.load(dp, allow_pickle=False) as f:
        foot, groups, ids, times = (
            f[k].copy() for k in ("features", "context", "condition", "control")
        )
    with np.load(hp, allow_pickle=False) as f:
        history = f["features"].copy()
    n = continuous["samples"]
    if (
        history.shape != (n, 308)
        or foot.shape != (n, 12)
        or any(v.shape != (n,) for v in (groups, ids, times))
        or not np.isfinite(history).all()
        or not np.array_equal(history[:, :12], foot.astype(np.float32))
    ):
        raise ValueError("all exact original foot features and finite history required")
    originals = {(r["context"], r["side"]): (i, r) for i, r in enumerate(continuous["records"])}
    histories = {(r["context"], r["side"]): r for r in receipt["records"]}
    pins = {(r["block"], r["expanded"]): r for r in dataset["source_pins"]}
    if len(originals) != 72 or len(histories) != 72 or len(pins) != 6:
        raise ValueError("all distinct source conditions required")
    used, seen, blocks, frozen = set(), set(), set(), []
    for root_name, audit_name in a.pair:
        root, own = Path(root_name), Path(audit_name)
        decpath = root / "declaration.json"
        dec = json.loads(decpath.read_text())
        expanded = dec["protocol"] == "soccer.neural_relay_balance_expanded.development.v6"
        if not expanded and dec["protocol"] != "soccer.neural_relay_balance_teacher.development.v5":
            raise ValueError("original study protocol required")
        key = (dec["development_context_block"], expanded)
        if key in blocks or key not in pins:
            raise ValueError("unique audited blocks required")
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
            raise ValueError("original independently replayed source required")
        for trial in source["records"]:
            if trial["course"]["arm"]["balance_teacher"] != -1:
                continue
            condition = (trial["course"]["context"], trial["course"]["lateral_m"])
            if condition not in originals or condition not in histories or condition in seen:
                raise ValueError("unique source condition required")
            seen.add(condition)
            index, row = originals[condition]
            hrow = histories[condition]
            start, count = row["first_sample"], row["sample_count"]
            if (
                hrow["block"] != key[0]
                or hrow["expanded"] != expanded
                or hrow["source_trial"] != trial["index"]
                or hrow["first_sample"] != start
                or hrow["sample_count"] != count
                or start < 0
                or count < 1
                or start + count > n
                or not np.array_equal(ids[start : start + count], np.full(count, index))
                or not np.array_equal(groups[start : start + count], np.full(count, condition[0]))
                or not np.array_equal(times[start : start + count], np.arange(count))
            ):
                raise ValueError("actual original causal interval required")
            velocity = bind(root, trial, "receiver_velocity")
            if velocity is None or digest(velocity) != hrow["velocity_hash"]:
                raise ValueError("exact own motor history required")
            with np.load(velocity, allow_pickle=False) as f:
                obs, control = f["obs"], f["control"]
                if (
                    obs.ndim != 2
                    or obs.shape[1] != 98
                    or len(obs) < count
                    or obs.dtype != np.float32
                    or not np.isfinite(obs).all()
                    or not np.array_equal(control, np.arange(len(obs)))
                ):
                    raise ValueError("exact consecutive own observations required")
                segment = history[start : start + count]
                if (
                    not np.array_equal(segment[:, 12:110], obs[:count])
                    or not np.array_equal(segment[1:, 110:208], obs[: count - 1])
                    or not np.array_equal(segment[2:, 208:306], obs[: max(0, count - 2)])
                    or np.any(segment[0, 110:208])
                    or np.any(segment[:2, 208:306])
                    or not np.array_equal(segment[:, 306], np.arange(count) >= 1)
                    or not np.array_equal(segment[:, 307], np.arange(count) >= 2)
                ):
                    raise ValueError("current/past motor observations or padding differ")
            for i in range(start, start + count):
                if i in used:
                    raise ValueError("sample overlap forbidden")
                used.add(i)
        frozen.extend(((root / "complete.json", sh), (own, pins[key]["audit_hash"])))
    if (
        blocks != set(pins)
        or seen != set(originals)
        or used != set(range(n))
        or any(digest(path) != expected for path, expected in checks)
        or digest(dp) != continuous["data_hash"]
        or digest(hp) != receipt["feature_hash"]
        or any(digest(path) != h for path, h in frozen)
    ):
        raise ValueError("complete unchanged data provenance required")
    clean(a.expected_head)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    write_once(
        a.output,
        dict(
            status="RELAY_CONTACT_HISTORY_READBACK_NOT_CONTROL",
            source_head=a.expected_head,
            features_hash=a.features_hash,
            continuous_hash=a.continuous_hash,
            conditions=72,
            samples=n,
            independently_rebuilt_causal_history=True,
            independent_physics_reintegration=False,
            promotion_authorized=False,
            hardware_authorized=False,
        ),
    )


if __name__ == "__main__":
    main()
