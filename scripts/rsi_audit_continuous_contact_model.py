"""Independent NumPy inference and condition-level forecast analysis."""

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_diagnose_pass_contact_timing import digest
from scripts.rsi_native_fresh_common import clean


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "features", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    for name in ("source-hash", "features-hash", "producer-head", "expected-head"):
        p.add_argument("--" + name, required=True)
    a = p.parse_args()
    clean(a.expected_head)
    sp, fp, dp = (
        a.source / "complete.json",
        a.features / "complete.json",
        a.features / "continuous.npz",
    )
    if (
        a.output.exists()
        or a.output.resolve().is_relative_to(Path.cwd())
        or any(p.is_symlink() for p in (sp, fp, dp))
        or digest(sp) != a.source_hash
        or digest(fp) != a.features_hash
    ):
        raise ValueError("exact trained model and continuous features required")
    source, receipt = json.loads(sp.read_text()), json.loads(fp.read_text())
    if (
        source.get("status") != "RELAY_CONTINUOUS_CONTACT_FORECAST_NOT_CONTROL"
        or source.get("source_head") != a.producer_head
        or source.get("features_hash") != a.features_hash
        or digest(dp) != receipt["data_hash"]
        or any(
            source.get(k) is not False
            for k in (
                "physical_control_authorized",
                "online_rl",
                "independent_physics_evaluation",
                "promotion_authorized",
                "hardware_authorized",
            )
        )
    ):
        raise ValueError("exact offline unpromoted grouped model required")
    with np.load(dp, allow_pickle=False) as f:
        x = f["features"].astype(np.float32)
        y, group, condition, controls = (
            f[k].copy() for k in ("labels", "context", "condition", "control")
        )
    if (
        x.shape != (receipt["samples"], 12)
        or any(z.shape != (len(x),) for z in (y, group, condition, controls))
        or not np.isfinite(x).all()
        or not np.isin(y, (0, 1, 2)).all()
    ):
        raise ValueError("finite complete continuous states required")
    contexts = sorted(int(c) for c in set(group.tolist()))
    rows = source["rows"]
    expected = {(c, s) for c in contexts for s in (17, 43, 71)}
    if (
        source["contexts"] != contexts
        or len(rows) != len(expected)
        or {(r["context"], r["seed"]) for r in rows} != expected
        or source["optimizer_updates"] != 200 * len(rows)
    ):
        raise ValueError("all grouped folds and predeclared updates required")
    predicted = {s: np.full(len(x), -1, dtype=np.int8) for s in (17, 43, 71)}
    max_error = 0.0
    for row in rows:
        train, test = group != row["context"], group == row["context"]
        cp = a.source / row["checkpoint"]
        if (
            cp.is_symlink()
            or cp.resolve().parent != a.source.resolve()
            or digest(cp) != row["checkpoint_hash"]
        ):
            raise ValueError("exact confined checkpoint required")
        with np.load(cp, allow_pickle=False) as f:
            center, scale, weights = f["center"], f["scale"], f["class_weights"]
            w1, b1, w2, b2 = f["weight1"], f["bias1"], f["weight2"], f["bias2"]
            indices, saved = f["heldout_indices"], f["heldout_logits"]
            counts = np.bincount(y[train], minlength=3)
            if (
                counts.shape != (3,)
                or (counts == 0).any()
                or center.shape != (12,)
                or scale.shape != (12,)
                or weights.shape != (3,)
                or w1.shape != (32, 12)
                or b1.shape != (32,)
                or w2.shape != (3, 32)
                or b2.shape != (3,)
                or any(
                    not np.isfinite(v).all()
                    for v in (center, scale, weights, w1, b1, w2, b2, saved)
                )
                or not np.array_equal(center, x[train].mean(axis=0))
                or not np.array_equal(scale, np.maximum(x[train].std(axis=0), 0.05))
                or not np.array_equal(indices, np.flatnonzero(test))
                or saved.shape != (int(test.sum()), 3)
            ):
                raise ValueError("exact training fold and finite checkpoint required")
            actual_weights = np.sqrt(int(train.sum()) / (3 * counts)).astype(np.float32)
            actual_weights /= actual_weights.mean()
            if not np.array_equal(weights, actual_weights):
                raise ValueError("class weights must use train fold only")
            z = np.clip((x[test] - center) / scale, -8, 8).astype(np.float32)
            logits = np.tanh(z @ w1.T + b1) @ w2.T + b2
        error = float(np.max(np.abs(logits - saved)))
        max_error = max(max_error, error)
        if error > 1e-5 or not np.array_equal(logits.argmax(axis=1), saved.argmax(axis=1)):
            raise ValueError("independent logits or decision differs")
        seed = row["seed"]
        if (predicted[seed][test] != -1).any():
            raise ValueError("overlapping holdouts forbidden")
        predicted[seed][test] = logits.argmax(axis=1)
        cm = np.zeros((3, 3), np.int64)
        np.add.at(cm, (y[test], predicted[seed][test]), 1)
        if cm.tolist() != row["confusion"]:
            raise ValueError("held-out confusion differs")
    summaries, per_condition = {}, []
    for seed, pred in predicted.items():
        if (pred < 0).any():
            raise ValueError("every held-out window must be predicted")
        cm = np.zeros((3, 3), np.int64)
        np.add.at(cm, (y, pred), 1)
        condition_counts = dict(
            nonfoot_conditions=0,
            nonfoot_detected=0,
            foot_conditions=0,
            foot_detected=0,
            no_contact_conditions=0,
            no_contact_conditions_with_false_alarm=0,
        )
        for idx in sorted(set(condition.tolist())):
            mask = condition == idx
            nonfoot = bool((y[mask] == 2).any())
            foot = bool((y[mask] == 1).any())
            no_contact = not nonfoot and not foot
            n_detected = bool(((y[mask] == 2) & (pred[mask] == 2)).any())
            f_detected = bool(((y[mask] == 1) & (pred[mask] == 1)).any())
            false_alarm = bool(((y[mask] == 0) & (pred[mask] != 0)).any())
            condition_counts["nonfoot_conditions"] += nonfoot
            condition_counts["nonfoot_detected"] += n_detected
            condition_counts["foot_conditions"] += foot
            condition_counts["foot_detected"] += f_detected
            condition_counts["no_contact_conditions"] += no_contact
            condition_counts["no_contact_conditions_with_false_alarm"] += no_contact and false_alarm
            per_condition.append(
                dict(
                    seed=seed,
                    condition=int(idx),
                    nonfoot=nonfoot,
                    nonfoot_detected=n_detected,
                    foot=foot,
                    foot_detected=f_detected,
                    no_contact=no_contact,
                    false_alarm=false_alarm,
                )
            )
        summaries[str(seed)] = dict(confusion=cm.tolist(), **condition_counts)
    if (
        digest(sp) != a.source_hash
        or digest(fp) != a.features_hash
        or digest(dp) != receipt["data_hash"]
    ):
        raise ValueError("input changed during independent inference")
    clean(a.expected_head)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    write_once(
        a.output,
        dict(
            status="RELAY_CONTINUOUS_CONTACT_NUMPY_AUDITED_NOT_CONTROL",
            source_head=a.expected_head,
            source_hash=a.source_hash,
            features_hash=a.features_hash,
            summaries=summaries,
            per_condition=per_condition,
            max_logit_error=max_error,
            condition_detection_is_not_independent_new_physics=True,
            physical_control_authorized=False,
            independent_physics_evaluation=False,
            promotion_authorized=False,
            hardware_authorized=False,
        ),
    )


if __name__ == "__main__":
    main()
