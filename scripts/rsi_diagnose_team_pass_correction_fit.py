"""Same-state weighted motor fit before/after correction, NOT physical success.

The before model is75024: its already CPU-audited177 original predictions,
its actual17 own-audited unassisted anchor actions, and its80 private shadow
predictions on the actual expert-continuation stream. No teacher h is reused.
Consumes a CLOSED independent CPU audit of the corrected model first.
"""

import argparse
import json
from pathlib import Path


def weighted_motor_mse(mean, data):
    import numpy as np

    target, scale, w = (data[k] for k in ("targets", "action_scale", "loss_weights"))
    if (
        mean.shape != target.shape
        or target.ndim != 2
        or target.shape[1] != 29
        or not 1 <= len(target) <= 400
        or scale.shape != (29,)
        or w.shape != (len(target),)
        or any(not np.isfinite(x).all() for x in (mean, target, scale, w))
        or (w <= 0).any()
        or (scale <= 0).any()
    ):
        raise ValueError("whole finite actual motor history and positive measured weights required")
    error = ((mean - target) * scale).astype(np.float64) ** 2
    w = w.astype(np.float64)
    return float((error.mean(axis=1) * w).sum() / w.sum())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("training-root", "cpu-own-root", "output"):
        p.add_argument("--" + k, type=Path, required=True)
    for k in ("training-hash", "cpu-own-hash", "expected-head"):
        p.add_argument("--" + k, required=True)
    a = p.parse_args()
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest
    from scripts.rsi_native_fresh_common import clean, receipt, source_pins

    clean(a.expected_head)
    paths = [a.training_root / "complete.json", a.cpu_own_root / "complete.json"]
    if (
        a.output.exists()
        or [digest(x) for x in paths] != [a.training_hash, a.cpu_own_hash]
        or any(x.is_symlink() or (x.parent / "failed.json").exists() for x in paths)
    ):
        raise ValueError("exact trained model and closed independent CPU proof required")
    trained, own = [json.loads(x.read_text()) for x in paths]
    if (
        own["status"] != "DUAL_RECURRENT_TRAINED_CHECKPOINT_CPU_OWN_HISTORY_AUDIT_PASS_NOT_PHYSICS"
        or own["checkpoint_kind"] != "team-pass-corrected-v2"
        or own["source_hash"] != a.training_hash
        or own["checkpoint_hash"] != trained["checkpoint_hash"]
        or own["own_actor_decisions"] != trained["cpu_offline_decisions"]
    ):
        raise ValueError("all274 own-history independent corrected-model predictions required")
    pins = dict(own["input_file_hashes"])
    pins.update({str(x): digest(x) for x in paths})
    bp, bank = receipt(pins, trained["data_hash"])
    wp, warm = receipt(pins, trained["warm_start_hash"])
    original_path, original = receipt(pins, bank["base_data_hash"])
    if (
        len(bank["records"]) != 274
        or len(original["records"]) != 177
        or warm["data_hash"] != bank["base_data_hash"]
        or warm["config"]["seed"] != 75024
    ):
        raise ValueError("same original/current/correction histories and research student required")
    proofs = []
    for path in pins:
        if path.endswith("/complete.json"):
            d = json.loads(Path(path).read_text())
            if (
                d.get("status")
                == "DUAL_RECURRENT_TRAINED_CHECKPOINT_CPU_OWN_HISTORY_AUDIT_PASS_NOT_PHYSICS"
                and d.get("source_hash") == trained["warm_start_hash"]
            ):
                proofs.append(d)
    if (
        len(proofs) != 1
        or proofs[0]["own_actor_decisions"] != warm["cpu_offline_decisions"]
        or proofs[0]["checkpoint_hash"] != warm["checkpoint_hash"]
        or proofs[0]["checkpoint_kind"] != "team-pass-consolidated-v1"
        or proofs[0]["data_hash"] != bank["base_data_hash"]
    ):
        raise ValueError("exact whole177 independent CPU before-model proof required")
    import numpy as np

    def load(path, h):
        if path.is_symlink() or digest(path) != h:
            raise ValueError("exact actual data or independently replayed prediction required")
        pins[str(path)] = h
        with np.load(path, allow_pickle=False) as z:
            return {k: z[k].copy() for k in z.files}

    groups = {}
    for row, prediction in zip(bank["records"], trained["cpu_replay"], strict=True):
        if not row["task_success"]:
            continue
        data = load(bp.parent / row["raw_path"], row["raw_hash"])
        after = load(
            a.training_root / prediction["prediction_path"], prediction["prediction_hash"]
        )["means"]
        i = row["index"]
        if i < 177:
            raw = original["records"][i]
            olddata = load(original_path.parent / raw["raw_path"], raw["raw_hash"])
            if any(not np.array_equal(data[k], olddata[k]) for k in ("observations", "targets")):
                raise ValueError("same-state before prediction must use identical entire history")
            pred = warm["cpu_replay"][i]
            before = load(wp.parent / pred["prediction_path"], pred["prediction_hash"])["means"]
        elif i < 194:
            before = data["targets"].copy()
        else:
            path = Path(row["source_root"]) / f"trial{row['source_index']}/A.npz"
            saved = load(path, pins[str(path)])
            if not np.array_equal(saved["NN_obs"], data["observations"]):
                raise ValueError("actual student shadow history differs from correction sequence")
            before = saved["STUDENT_NN_action"]
        if before.shape != after.shape or before.shape != data["targets"].shape:
            raise ValueError("whole before/after motor histories required")

        groups.setdefault(row["role"], []).append(
            dict(
                index=i,
                before=weighted_motor_mse(before, data),
                after=weighted_motor_mse(after, data),
            )
        )
    summary = {}
    for role, rows in groups.items():
        before = float(np.mean([x["before"] for x in rows]))
        after = float(np.mean([x["after"] for x in rows]))
        summary[role] = dict(
            positive_histories=len(rows),
            before_weighted_MSE_rad2=before,
            after_weighted_MSE_rad2=after,
            after_before_ratio=None if before == 0 else after / before,
        )
    source_pins(pins)
    check_pins(pins)
    clean(a.expected_head)
    a.output.mkdir()
    write_once(
        a.output / "complete.json",
        dict(
            status="SAME_ACTUAL_HISTORY_CORRECTION_FIT_DIAGNOSTIC_NOT_PHYSICAL_SUCCESS",
            source_head=a.expected_head,
            input_file_hashes=pins,
            training_hash=a.training_hash,
            cpu_own_hash=a.cpu_own_hash,
            summary=summary,
            rows=groups,
            positive_histories=sum(len(rows) for rows in groups.values()),
            physical_steps=0,
            optimizer_updates=0,
            student_success_claim=False,
            goal_complete=False,
            hardware_authorized=False,
            promotion_authorized=False,
        ),
    )


if __name__ == "__main__":
    main()
