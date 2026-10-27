"""Independent NumPy inference audit for grouped contact-event classifiers."""

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
        a.features / "foot-entry.npz",
    )
    if (
        a.output.exists()
        or a.output.resolve().is_relative_to(Path.cwd())
        or any(x.is_symlink() for x in (sp, fp, dp))
        or digest(sp) != a.source_hash
        or digest(fp) != a.features_hash
    ):
        raise ValueError("exact trained source and feature receipt required")
    source, receipt = json.loads(sp.read_text()), json.loads(fp.read_text())
    if (
        source.get("status") != "RELAY_FOOT_CONTACT_EVENT_CLASSIFIER_NOT_CONTROL"
        or source.get("source_head") != a.producer_head
        or source.get("features_hash") != a.features_hash
        or source.get("retrospective_contact_condition") is not True
        or digest(dp) != receipt["feature_hash"]
        or source.get("threshold") != 0.5
        or any(
            source.get(k) is not False
            for k in (
                "physical_control_authorized",
                "independent_physics_evaluation",
                "promotion_authorized",
                "hardware_authorized",
            )
        )
    ):
        raise ValueError("exact unpromoted event-conditioned model required")
    with np.load(dp, allow_pickle=False) as f:
        x = f["foot_relative"].reshape(-1, 12).astype(np.float32)
    labels, groups = np.empty(len(x), dtype=np.int64), np.empty(len(x), dtype=np.int64)
    indexes = set()
    for item in receipt["records"]:
        if not item["receiver_contact"]:
            if item["feature_index"] is not None:
                raise ValueError("no-contact data incorrectly entered training")
            continue
        index, cats = item["feature_index"], item["first_contact_force_categories"]
        if type(index) is not int or index in indexes or not 0 <= index < len(x) or not cats:
            raise ValueError("unique first-contact label required")
        indexes.add(index)
        labels[index], groups[index] = int(2 in cats), item["context"]
    if indexes != set(range(len(x))):
        raise ValueError("every first-contact feature must have an observed label")
    contexts = sorted(int(x) for x in set(groups.tolist()))
    expected = {(c, s) for c in contexts for s in (17, 43, 71)}
    rows = source["rows"]
    if (
        indexes != set(range(len(x)))
        or source["contexts"] != contexts
        or len(rows) != len(expected)
        or {(r["context"], r["seed"]) for r in rows} != expected
        or source["optimizer_updates"] != len(rows) * 300
    ):
        raise ValueError("all grouped folds, seeds and updates required")
    confusion = {str(s): dict(tp=0, tn=0, fp=0, fn=0) for s in (17, 43, 71)}
    max_error = 0.0
    for row in rows:
        train, test = groups != row["context"], groups == row["context"]
        cp = a.source / row["checkpoint"]
        if (
            cp.is_symlink()
            or cp.resolve().parent != a.source.resolve()
            or digest(cp) != row["checkpoint_hash"]
            or row["training_samples"] != int(train.sum())
        ):
            raise ValueError("exact confined checkpoint and group split required")
        with np.load(cp, allow_pickle=False) as f:
            center, scale = f["center"], f["scale"]
            w1, b1, w2, b2 = f["weight1"], f["bias1"], f["weight2"], f["bias2"]
            if (
                center.shape != (12,)
                or scale.shape != (12,)
                or w1.shape != (32, 12)
                or b1.shape != (32,)
                or w2.shape != (1, 32)
                or b2.shape != (1,)
                or any(not np.isfinite(v).all() for v in (center, scale, w1, b1, w2, b2))
                or not np.array_equal(center, x[train].mean(axis=0))
                or not np.array_equal(scale, np.maximum(x[train].std(axis=0), 0.05))
            ):
                raise ValueError("exact train-only normalizer and finite network required")
            z = np.clip((x[test] - center) / scale, -8, 8).astype(np.float32)
            logit = (np.tanh(z @ w1.T + b1) @ w2.T + b2).reshape(-1)
            scores = 1 / (1 + np.exp(-logit))
        indices = np.flatnonzero(test)
        if len(indices) != len(row["heldout"]):
            raise ValueError("complete grouped holdout required")
        for i, score, saved in zip(indices, scores, row["heldout"], strict=True):
            error = abs(float(score) - saved["score_nonfoot"])
            max_error = max(max_error, error)
            prediction = bool(score >= 0.5)
            target = bool(labels[i])
            if (
                saved["index"] != int(i)
                or saved["target_nonfoot"] != target
                or saved["prediction_nonfoot"] != prediction
                or error > 1e-6
            ):
                raise ValueError("independent NumPy prediction disagrees")
            key = (
                "tp" if target and prediction else "fn" if target else "fp" if prediction else "tn"
            )
            confusion[str(row["seed"])][key] += 1
    if (
        digest(sp) != a.source_hash
        or digest(fp) != a.features_hash
        or digest(dp) != receipt["feature_hash"]
    ):
        raise ValueError("classifier inputs changed during audit")
    clean(a.expected_head)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    write_once(
        a.output,
        dict(
            status="RELAY_FOOT_EVENT_NUMPY_AUDITED_NOT_CONTROL",
            source_head=a.expected_head,
            source_hash=a.source_hash,
            features_hash=a.features_hash,
            confusion=confusion,
            max_score_error=max_error,
            retrospective_contact_condition=True,
            independent_physics_evaluation=False,
            physical_control_authorized=False,
            promotion_authorized=False,
            hardware_authorized=False,
        ),
    )


if __name__ == "__main__":
    main()
