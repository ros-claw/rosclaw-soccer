"""Independently recompute held-out warning logits and calibration in NumPy."""

import argparse
import json
from pathlib import Path

import numpy as np

from rosclaw_soccer.rsi.condition_hazard_calibration import (
    calibrate_negative_maxima,
    condition_hazard_summary,
)
from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_diagnose_pass_contact_timing import digest
from scripts.rsi_native_fresh_common import clean


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("features", "training", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    for name in ("features-hash", "training-hash", "expected-head"):
        p.add_argument("--" + name, required=True)
    a = p.parse_args()
    clean(a.expected_head)
    manifest, raw, receipt = (
        a.features / "complete.json",
        a.features / "continuous.npz",
        a.training / "complete.json",
    )
    if (
        a.output.exists()
        or a.output.resolve().is_relative_to(Path.cwd())
        or any(p.is_symlink() for p in (manifest, raw, receipt))
        or digest(manifest) != a.features_hash
        or digest(receipt) != a.training_hash
    ):
        raise ValueError("exact original source, trained models and unused audit required")
    data, training = json.loads(manifest.read_text()), json.loads(receipt.read_text())
    if (
        training.get("status") != "CALIBRATED_CONTACT_WARNING_GROUP_HELDOUT_NOT_CONTROL"
        or training.get("features_hash") != a.features_hash
        or len(training.get("rows", [])) != 36
        or digest(raw) != data["data_hash"]
        or any(
            training.get(k) is not False
            for k in (
                "physical_control_authorized",
                "independent_new_physics",
                "online_rl",
                "promotion_authorized",
                "hardware_authorized",
            )
        )
    ):
        raise ValueError("complete unpromoted cross-validation forecast required")
    with np.load(raw, allow_pickle=False) as f:
        x = f["features"].astype(np.float32)
        labels, context, condition = (f[k].copy() for k in ("labels", "context", "condition"))
    contexts = sorted(int(c) for c in set(context.tolist()))
    if len(contexts) != 12:
        raise ValueError("exact 12 context groups required")
    summaries, frozen, seen, max_error = {}, [], set(), 0.0
    for row in training["rows"]:
        heldout, seed = row["heldout_context"], row["seed"]
        key = (heldout, seed)
        if key in seen or heldout not in contexts or seed not in (17, 43, 71):
            raise ValueError("unique heldout context and seed required")
        seen.add(key)
        idx = contexts.index(heldout)
        cal = [contexts[(idx + i) % len(contexts)] for i in (1, 2, 3, 4)]
        train = [c for c in contexts if c != heldout and c not in cal]
        if row["calibration_contexts"] != cal or row["training_contexts"] != train:
            raise ValueError("strictly separated training, calibration and test groups required")
        path = a.training / row["checkpoint"]
        if (
            path.is_symlink()
            or path.resolve().parent != a.training.resolve()
            or digest(path) != row["checkpoint_hash"]
        ):
            raise ValueError("exact confined heldout model required")
        frozen.append((path, row["checkpoint_hash"]))
        with np.load(path, allow_pickle=False) as cp:
            train_mask, cal_mask, test = (
                np.isin(context, train),
                np.isin(context, cal),
                context == heldout,
            )
            center = x[train_mask].mean(axis=0)
            scale = np.maximum(x[train_mask].std(axis=0), 0.05)
            if not np.array_equal(cp["center"], center) or not np.array_equal(cp["scale"], scale):
                raise ValueError("normalization must use only training contexts")
            z = np.clip((x - center) / scale, -8, 8).astype(np.float32)
            logits = np.tanh(z @ cp["weight1"].T + cp["bias1"]) @ cp["weight2"].T + cp["bias2"]
            if not np.array_equal(cp["heldout_indices"], np.flatnonzero(test)):
                raise ValueError("exact independent heldout indices required")
            error = float(np.max(np.abs(logits[test] - cp["heldout_logits"])))
            if not np.isfinite(error) or error > 2e-5:
                raise ValueError("NumPy and trained heldout logits differ")
            max_error = max(max_error, error)
            shift = logits - logits.max(axis=1, keepdims=True)
            exp = np.exp(shift)
            risk = (exp / exp.sum(axis=1, keepdims=True))[:, 2]
            cal_rows = condition_hazard_summary(
                risk[cal_mask], labels[cal_mask], condition[cal_mask]
            )
            threshold, negatives = calibrate_negative_maxima(cal_rows, alpha=0.25)
            if (
                not np.isclose(threshold, row["threshold"], atol=2e-6, rtol=0)
                or negatives != row["calibration_negative_conditions"]
                or (threshold == 1.0) != row["insufficient_calibration"]
            ):
                raise ValueError("training-only condition calibration differs")
            actual = condition_hazard_summary(risk[test], labels[test], condition[test])
            if len(actual) != len(row["test_conditions"]):
                raise ValueError("all heldout condition outputs required")
            for got, expected in zip(actual, row["test_conditions"], strict=True):
                if (
                    got["condition"] != expected["condition"]
                    or got["contains_nonfoot"] != expected["contains_nonfoot"]
                    or not np.isclose(got["max_risk"], expected["max_risk"], atol=2e-6, rtol=0)
                    or (got["max_risk"] > threshold) != expected["alarm"]
                ):
                    raise ValueError("heldout condition warning differs")
                got["alarm"] = expected["alarm"]
            summaries.setdefault(str(seed), []).extend(actual)
    if seen != {(context, seed) for context in contexts for seed in (17, 43, 71)}:
        raise ValueError("complete heldout folds and seeds required")
    aggregates = {}
    for seed, rows in summaries.items():
        pos, neg = (
            [r for r in rows if r["contains_nonfoot"]],
            [r for r in rows if not r["contains_nonfoot"]],
        )
        aggregates[seed] = dict(
            conditions=len(rows),
            positive_conditions=len(pos),
            positive_detected=sum(r["alarm"] for r in pos),
            negative_conditions=len(neg),
            negative_with_alarm=sum(r["alarm"] for r in neg),
            folds_without_calibration=sum(
                r["insufficient_calibration"] for r in training["rows"] if r["seed"] == int(seed)
            ),
        )
    if (
        aggregates != training["summaries"]
        or digest(manifest) != a.features_hash
        or digest(receipt) != a.training_hash
        or digest(raw) != data["data_hash"]
        or any(digest(path) != h for path, h in frozen)
    ):
        raise ValueError("recomputed counts or immutable evidence differ")
    clean(a.expected_head)
    a.output.mkdir(parents=True)
    write_once(
        a.output / "complete.json",
        dict(
            status="CALIBRATED_CONTACT_WARNING_NUMPY_READBACK_NOT_CONTROL",
            source_head=a.expected_head,
            training_hash=a.training_hash,
            features_hash=a.features_hash,
            max_logit_error=max_error,
            summaries=aggregates,
            calibration_uses_training_only_groups=True,
            independent_new_physics=False,
            holdout_is_within_development_source=True,
            physical_control_authorized=False,
            promotion_authorized=False,
            hardware_authorized=False,
            goal_complete=False,
        ),
    )
    print(json.dumps(aggregates))


if __name__ == "__main__":
    main()
