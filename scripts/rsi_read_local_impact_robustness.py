"""Whole local24 raw outcome/decision readback, no new physics or learning.

Scoring and comparison formulas are shared. This is not an independent neural
or physical replay and cannot qualify a teacher or admit a learning dataset.
"""

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np

from rosclaw_soccer.providers.g1.sonic_runup import G1SonicRunupController
from rosclaw_soccer.rsi.first_touch_outcome import label_first_touch
from rosclaw_soccer.rsi.foot_heading_parity import original_arrays
from rosclaw_soccer.rsi.local_impact_perturbation_plan import declare_local_impact_perturbations
from rosclaw_soccer.rsi.native_first_touch_initial_state import NativeFirstTouchInitialState
from rosclaw_soccer.sim.contracts import G1_HARD_TORQUE_LIMITS, hash_json
from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_early_episode_decisions import read_decisions
from scripts.rsi_early_impact_execution_alignment import verify_actual_torques
from scripts.rsi_early_impact_h300 import intervention_frames
from scripts.rsi_local_impact_robustness import paired_summary, require_scope
from scripts.rsi_native_prospective_impact import require_prefix
from scripts.rsi_stage_release_contact_replay import digest, require_terminal


def verify_original_control_torques(trace):
    """Check zero additional control directly; do not invent early-action fields."""
    original_arrays(trace)
    expected = np.clip(
        trace["pd_requested_torque_nm"],
        -np.asarray(G1_HARD_TORQUE_LIMITS),
        np.asarray(G1_HARD_TORQUE_LIMITS),
    )
    if any(
        not np.array_equal(trace[k], expected) for k in ("torque_nm", "actual_actuator_force_nm")
    ):
        raise ValueError("original-control actual torques differ from clipped base PD")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--unit", required=True)
    parser.add_argument("--source-head", required=True)
    parser.add_argument("--complete-file-hash", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    source = Path(__file__).resolve().parent.parent
    if (
        args.output.exists()
        or (args.root / "failure.json").exists()
        or digest(args.root / "complete.json") != args.complete_file_hash
        or subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
        != args.source_head
        or subprocess.check_output(["git", "status", "--porcelain"], cwd=source, text=True).strip()
    ):
        raise ValueError("whole terminal bytes, clean frozen reader and unused output required")
    require_terminal(args.unit)
    pins = {
        str(p): digest(p)
        for folder in (source / "src", source / "scripts")
        for p in folder.rglob("*.py")
    }

    def pin(name, expected=None):
        p = Path(name)
        h = digest(p)
        if (
            p.is_symlink()
            or (expected is not None and h != expected)
            or (str(p) in pins and pins[str(p)] != h)
        ):
            raise ValueError("same ordinary complete local24 source/input bytes required")
        pins[str(p)] = h

    def sealed(name):
        pin(name)
        d = json.loads(Path(name).read_text())
        if d["report_hash"] != hash_json({k: v for k, v in d.items() if k != "report_hash"}):
            raise ValueError("whole local24 logical seal required")
        return d

    def arrays(w):
        pin(w["path"], w["sha256"])
        with np.load(w["path"], allow_pickle=False) as a:
            return dict(a)

    c, d = sealed(args.root / "complete.json"), sealed(args.root / "declaration.json")
    cfg = d["config"]
    require_scope(cfg)
    if (
        c["schema"] != "soccer.rsi.local_impact_robustness_complete.v1"
        or c["declaration_hash"] != d["report_hash"]
        or Path(cfg["output_root"]) != args.root
        or len(c["rows"]) != 12
        or type(c["new_native_steps"]) is not int
        or c["new_native_steps"] != 72000
        or type(c["new_optimizer_updates"]) is not int
        or c["new_optimizer_updates"] != 0
        or any(
            c[k] is not False
            for k in (
                "teacher_qualified",
                "training_authorized",
                "hardware_authorized",
                "promotion_authorized",
            )
        )
    ):
        raise ValueError("all original24 unqualified paired development executions required")
    for name, h in d["input_file_hashes"].items():
        pin(name, h)
    stage = sealed(cfg["stage_readback"])
    native = sealed(stage["effective_cohort_spec"]["producer_complete"])
    h300 = sealed(Path(cfg["producer_root"]) / "complete.json")
    if (
        len(native["results"]) != 6
        or len(h300["rows"]) != 6
        or h300["report_hash"] != d["reused_h300_complete_hash"]
    ):
        raise ValueError("whole original baseline and correction outcomes required")
    commitments = []
    for receipt in native["results"]:
        witness = receipt["witnesses"]["commitment"]
        # Native commitments have no embedded report_hash: authenticate their
        # full bytes against the original receipt instead of inventing a seal.
        pin(witness["path"], witness["sha256"])
        commitments.append(json.loads(Path(witness["path"]).read_text()))
    plan = declare_local_impact_perturbations(tuple(tuple(r["course"]) for r in commitments))
    if plan != d["plan"] or d["old_baseline_and_early_outcomes_reused"] != [
        dict(index=i, baseline=r["outcome"], early=h300["rows"][i]["causal_rollout_outcome"])
        for i, r in enumerate(native["results"])
    ]:
        raise ValueError(
            "same full outcome-independent local course plan and reused results required"
        )
    reviews = []
    forecast_total = 0
    extra = {
        "early_impact_feedback_" + s
        for s in ("requested_offset_nm", "applied_offset_nm", "forecast_steps")
    }
    for i, (row, job) in enumerate(zip(c["rows"], plan["jobs"], strict=True)):
        if (
            row != sealed(args.root / f"pair-{i}-complete.json")
            or type(row["index"]) is not int
            or row["index"] != i
            or row["job"] != job
            or row["original_prefix_matched"] is not True
        ):
            raise ValueError("all original ordered local pairs including failures required")
        if len(row["arms"]) != 2:
            raise ValueError("both original and correction arms required")
        original = None
        arm_reviews = []
        for j, arm in enumerate(("original_control", "prospective_early_impact")):
            a = row["arms"][j]
            enabled = bool(j)
            parent = native["results"][job["source_index"]]
            commitment = commitments[job["source_index"]]
            if (
                a != sealed(args.root / f"pair-{i}-{arm}-complete.json")
                or a["arm"] != arm
                or a["job"] != job
                or a["source_receipt_hash"] != parent["receipt_hash"]
                or a["original_policy_hash"] != commitment["motor_policy_hash"]
                or a["owned_continuation_enabled"] is not enabled
                or any(
                    a[k] is not False
                    for k in ("teacher_qualified", "hardware_authorized", "promotion_authorized")
                )
                or a["arrays"]["path"] != str(args.root / f"pair-{i}-{arm}-trace.npz")
            ):
                raise ValueError("same original policy, course and correctly ordered arms required")
            trace = arrays(a["arrays"])
            if enabled:
                if set(trace) != set(original) | extra:
                    raise ValueError(
                        "exact original capture plus complete correction fields required"
                    )
                original_arrays({k: v for k, v in trace.items() if k not in extra})
            else:
                verify_original_control_torques(trace)
            initial = NativeFirstTouchInitialState(
                tuple(job["course"]),
                np.asarray(G1SonicRunupController.default_angles, dtype=np.float64),
            )
            initial.verify(
                job["course"], trace["canonical_qpos"][0, 0], trace["canonical_qvel"][0, 0]
            )
            if a["review"]["initial_state_contract"] != initial.contract():
                raise ValueError("paired arm initial state differs from declared perturbed course")
            outcome = label_first_touch(trace).to_dict()
            if outcome != a["outcome"]:
                raise ValueError("whole raw M0 outcome differs")
            selected = []
            count = 0
            requested = applied = None
            if enabled:
                verify_actual_torques(trace)
                requested, applied = intervention_frames(trace)
                require_prefix(trace, original, applied)
                f = trace["early_impact_feedback_forecast_steps"]
                if f.shape != (300, 1, 1) or f.dtype != np.int64 or np.any(f < 0) or np.any(f % 40):
                    raise ValueError("all typed per-frame forecast counts required")
                count = int(np.sum(f))
                selected = read_decisions(
                    trace, args.root / f"pair-{i}-preaction-evidence", sealed=sealed, arrays=arrays
                )
            else:
                original = trace
            review = a["review"]
            if (
                a["first_requested_correction_frame"] != requested
                or a["first_physically_applied_correction_frame"] != applied
                or any(
                    type(review[k]) is not int or review[k] != n
                    for k, n in (
                        ("frames", 300),
                        ("native_attempted_steps", 3000),
                        ("native_integrated_steps", 3000),
                        ("forecast_attempted_steps", count),
                        ("forecast_integrated_steps", count),
                    )
                )
                or not 0 <= count <= 1800
                or review["current_body_policy_recomputed"] is not True
                or review["source_future_trace_or_targets_consumed"] is not False
            ):
                raise ValueError(
                    "same whole physical body/correction boundary and counters required"
                )
            forecast_total += count
            arm_reviews.append(
                dict(
                    arm=arm,
                    outcome=outcome,
                    selected_window_alignment=selected,
                    forecast_steps=count,
                )
            )
        reviews.append(dict(index=i, job=job, arms=arm_reviews))
    summary = paired_summary(c["rows"])
    if (
        any(type(c[k]) is not type(v) or c[k] != v for k, v in summary.items())
        or type(c["forecast_steps"]) is not int
        or c["forecast_steps"] != forecast_total
        or forecast_total > 21600
    ):
        raise ValueError("complete local24 task and execution summary differs")
    if any(digest(name) != h for name, h in pins.items()):
        raise ValueError("local24 source/input changed during readback")
    result = dict(
        schema="soccer.rsi.local_impact_robustness_readback.v1",
        source_head=args.source_head,
        complete_hash=c["report_hash"],
        input_file_hashes=pins,
        rows=reviews,
        **summary,
        actual_native_steps=72000,
        actual_forecast_steps=forecast_total,
        new_unique_physical_courses=6,
        original_source_clusters=3,
        all24_executions_including_failures_retained=True,
        new_physical_steps=0,
        new_neural_inferences=0,
        new_optimizer_updates=0,
        shared_label_and_score_algorithms=True,
        independent_physical_replay=False,
        independent_neural_implementation=False,
        hardware_authorized=False,
    )
    result["report_hash"] = hash_json(result)
    write_once(args.output, result)


if __name__ == "__main__":
    main()
