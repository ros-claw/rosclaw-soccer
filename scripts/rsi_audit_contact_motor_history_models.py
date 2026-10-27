"""Independent NumPy replay and matched parent comparison for history models."""

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_diagnose_pass_contact_timing import digest
from scripts.rsi_native_fresh_common import clean


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "history", "continuous", "parent", "parent-audit", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    for name in (
        "source-hash",
        "history-hash",
        "continuous-hash",
        "parent-hash",
        "parent-audit-hash",
        "producer-head",
        "expected-head",
    ):
        p.add_argument("--" + name, required=True)
    a = p.parse_args()
    clean(a.expected_head)
    sp, hp, cp, pp, pa = (
        a.source / "complete.json",
        a.history / "complete.json",
        a.continuous / "complete.json",
        a.parent / "complete.json",
        a.parent_audit / "complete.json",
    )
    checks = (
        (sp, a.source_hash),
        (hp, a.history_hash),
        (cp, a.continuous_hash),
        (pp, a.parent_hash),
        (pa, a.parent_audit_hash),
    )
    hdata, cdata = a.history / "motor-history.npz", a.continuous / "continuous.npz"
    if (
        a.output.exists()
        or a.output.resolve().is_relative_to(Path.cwd())
        or any(path.is_symlink() or digest(path) != expected for path, expected in checks)
        or hdata.is_symlink()
        or cdata.is_symlink()
    ):
        raise ValueError("exact matched model and data, unused external output required")
    source, history, continuous, parent, parent_audit = (
        json.loads(path.read_text()) for path, _ in checks
    )
    if (
        source.get("status") != "RELAY_CONTACT_HISTORY_FORECAST_NOT_CONTROL"
        or source.get("source_head") != a.producer_head
        or source.get("history_hash") != a.history_hash
        or source.get("continuous_hash") != a.continuous_hash
        or source.get("parent_hash") != a.parent_hash
        or source.get("parent_audit_hash") != a.parent_audit_hash
        or parent_audit.get("status") != "RELAY_CONTINUOUS_CONTACT_NUMPY_AUDITED_NOT_CONTROL"
        or parent_audit.get("source_hash") != a.parent_hash
        or digest(hdata) != history["feature_hash"]
        or digest(cdata) != continuous["data_hash"]
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
        raise ValueError("fully audited matched unpromoted research sources required")
    with np.load(hdata, allow_pickle=False) as f:
        x = f["features"].copy()
    with np.load(cdata, allow_pickle=False) as f:
        y, groups, condition = (f[k].copy() for k in ("labels", "context", "condition"))
    n = continuous["samples"]
    if (
        x.shape != (n, 308)
        or x.dtype != np.float32
        or any(v.shape != (n,) for v in (y, groups, condition))
        or not np.isfinite(x).all()
        or not np.isin(y, (0, 1, 2)).all()
    ):
        raise ValueError("complete aligned finite 308D source inputs required")
    contexts = sorted(int(c) for c in set(groups.tolist()))
    expected = {(c, s) for c in contexts for s in (17, 43, 71)}
    p_rows = {(r["context"], r["seed"]): r for r in parent["rows"]}
    rows = source["rows"]
    if (
        source["contexts"] != contexts
        or parent["contexts"] != contexts
        or len(rows) != len(expected)
        or {(r["context"], r["seed"]) for r in rows} != expected
        or set(p_rows) != expected
        or source["optimizer_updates"] != len(rows) * 200
    ):
        raise ValueError("all exact matched grouped folds required")
    preds = {s: np.full(n, -1, dtype=np.int8) for s in (17, 43, 71)}
    olds = {s: np.full(n, -1, dtype=np.int8) for s in (17, 43, 71)}
    max_error = 0.0
    for row in rows:
        context, seed = row["context"], row["seed"]
        train, test = groups != context, groups == context
        current = a.source / row["checkpoint"]
        prow = p_rows[(context, seed)]
        old = a.parent / prow["checkpoint"]
        if (
            current.is_symlink()
            or current.resolve().parent != a.source.resolve()
            or digest(current) != row["checkpoint_hash"]
            or old.is_symlink()
            or old.resolve().parent != a.parent.resolve()
            or digest(old) != prow["checkpoint_hash"]
            or row["parent_checkpoint_hash"] != prow["checkpoint_hash"]
        ):
            raise ValueError("exact paired confined checkpoints required")
        with np.load(current, allow_pickle=False) as f, np.load(old, allow_pickle=False) as o:
            center, scale, weights = f["center"], f["scale"], f["class_weights"]
            w1, b1, w2, b2 = f["weight1"], f["bias1"], f["weight2"], f["bias2"]
            indices, saved, old_indices, old_logits = (
                f["heldout_indices"],
                f["heldout_logits"],
                o["heldout_indices"],
                o["heldout_logits"],
            )
            counts = np.bincount(y[train], minlength=3)
            if (
                counts.shape != (3,)
                or (counts == 0).any()
                or center.shape != (308,)
                or scale.shape != (308,)
                or weights.shape != (3,)
                or w1.shape != (32, 308)
                or b1.shape != (32,)
                or w2.shape != (3, 32)
                or b2.shape != (3,)
                or saved.shape != (int(test.sum()), 3)
                or old_logits.shape != saved.shape
                or not np.array_equal(indices, np.flatnonzero(test))
                or not np.array_equal(indices, old_indices)
                or any(
                    not np.isfinite(v).all()
                    for v in (center, scale, weights, w1, b1, w2, b2, saved, old_logits)
                )
                or not np.array_equal(center, x[train].mean(axis=0))
                or not np.array_equal(scale, np.maximum(x[train].std(axis=0), 0.05))
            ):
                raise ValueError("exact finite matching train-only split required")
            true_weights = np.sqrt(int(train.sum()) / (3 * counts)).astype(np.float32)
            true_weights /= true_weights.mean()
            if not np.array_equal(weights, true_weights):
                raise ValueError("training-only class weights required")
            z = np.clip((x[test] - center) / scale, -8, 8).astype(np.float32)
            logits = np.tanh(z @ w1.T + b1) @ w2.T + b2
            old_pred = old_logits.argmax(axis=1)
        error = float(np.max(np.abs(logits - saved)))
        max_error = max(max_error, error)
        if error > 1e-4 or not np.array_equal(logits.argmax(axis=1), saved.argmax(axis=1)):
            raise ValueError("independent history inference differs")
        if (preds[seed][test] != -1).any() or (olds[seed][test] != -1).any():
            raise ValueError("overlapping holdouts forbidden")
        preds[seed][test], olds[seed][test] = logits.argmax(axis=1), old_pred
        cm = np.zeros((3, 3), dtype=np.int64)
        np.add.at(cm, (y[test], preds[seed][test]), 1)
        if cm.tolist() != row["confusion"]:
            raise ValueError("independently replayed fold confusion differs")
    summaries, detail = {}, []
    for seed in (17, 43, 71):
        now, old = preds[seed], olds[seed]
        if (now < 0).any() or (old < 0).any():
            raise ValueError("all held-out predictions required")
        cm = np.zeros((3, 3), dtype=np.int64)
        np.add.at(cm, (y, now), 1)
        stats = dict(
            confusion=cm.tolist(),
            nonfoot_window_rescues=int(((y == 2) & (old != 2) & (now == 2)).sum()),
            nonfoot_window_regressions=int(((y == 2) & (old == 2) & (now != 2)).sum()),
            false_alarm_window_rescues=int(((y == 0) & (old != 0) & (now == 0)).sum()),
            false_alarm_window_regressions=int(((y == 0) & (old == 0) & (now != 0)).sum()),
            no_contact_conditions=0,
            no_contact_old_false_alarm=0,
            no_contact_new_false_alarm=0,
            nonfoot_conditions=0,
            nonfoot_old_detected=0,
            nonfoot_new_detected=0,
        )
        for c in sorted(set(condition.tolist())):
            mask = condition == c
            none, nf = bool((y[mask] == 0).all()), bool((y[mask] == 2).any())
            ofp = bool(((y[mask] == 0) & (old[mask] != 0)).any())
            nfp = bool(((y[mask] == 0) & (now[mask] != 0)).any())
            od = bool(((y[mask] == 2) & (old[mask] == 2)).any())
            nd = bool(((y[mask] == 2) & (now[mask] == 2)).any())
            stats["no_contact_conditions"] += none
            stats["no_contact_old_false_alarm"] += none and ofp
            stats["no_contact_new_false_alarm"] += none and nfp
            stats["nonfoot_conditions"] += nf
            stats["nonfoot_old_detected"] += od
            stats["nonfoot_new_detected"] += nd
            detail.append(
                dict(
                    seed=seed,
                    condition=int(c),
                    no_contact=none,
                    nonfoot=nf,
                    old_false_alarm=ofp,
                    new_false_alarm=nfp,
                    old_nonfoot_detected=od,
                    new_nonfoot_detected=nd,
                )
            )
        summaries[str(seed)] = stats
    if (
        any(digest(path) != h for path, h in checks)
        or digest(hdata) != history["feature_hash"]
        or digest(cdata) != continuous["data_hash"]
    ):
        raise ValueError("source inputs changed during independent replay")
    clean(a.expected_head)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    write_once(
        a.output,
        dict(
            status="RELAY_CONTACT_HISTORY_PAIRED_NUMPY_NOT_CONTROL",
            source_head=a.expected_head,
            source_hash=a.source_hash,
            history_hash=a.history_hash,
            continuous_hash=a.continuous_hash,
            parent_hash=a.parent_hash,
            parent_audit_hash=a.parent_audit_hash,
            summaries=summaries,
            conditions=detail,
            max_logit_error=max_error,
            matched_original_conditions=True,
            physical_control_authorized=False,
            independent_physics_evaluation=False,
            promotion_authorized=False,
            hardware_authorized=False,
        ),
    )


if __name__ == "__main__":
    main()
