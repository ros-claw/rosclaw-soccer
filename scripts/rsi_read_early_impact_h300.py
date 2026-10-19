"""Whole-six H300 artifact and selected-window readback; no new physics/NN.

Labels and comparison formulas are shared, not independently implemented.
This never upgrades a development result to teacher/hardware qualification.
"""

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np

from rosclaw_soccer.rsi.first_touch_outcome import label_first_touch
from rosclaw_soccer.sim.contracts import hash_json
from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_early_episode_decisions import read_decisions
from scripts.rsi_early_impact_execution_alignment import verify_actual_torques
from scripts.rsi_early_impact_h300 import intervention_frames
from scripts.rsi_early_impact_pilot_inputs import require_scope
from scripts.rsi_native_prospective_impact import require_prefix
from scripts.rsi_stage_release_contact_replay import digest, require_terminal


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--unit", required=True)
    parser.add_argument("--source-head", required=True)
    parser.add_argument("--complete-file-hash", required=True)
    parser.add_argument("--output", type=Path, required=True)
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
        raise ValueError(
            "terminal whole pilot bytes, clean frozen reader and unused output required"
        )
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
            raise ValueError("whole original input bytes required")
        pins[str(p)] = h

    def sealed(name):
        pin(name)
        d = json.loads(Path(name).read_text())
        if d["report_hash"] != hash_json({k: v for k, v in d.items() if k != "report_hash"}):
            raise ValueError("whole logical seal required")
        return d

    def arrays(w):
        pin(w["path"], w["sha256"])
        with np.load(w["path"], allow_pickle=False) as a:
            return dict(a)

    complete = sealed(args.root / "complete.json")
    declaration = sealed(args.root / "declaration.json")
    cfg = declaration["config"]
    require_scope(cfg)
    if (
        complete["schema"] != "soccer.rsi.early_impact_h300_complete.v1"
        or complete["declaration_hash"] != declaration["report_hash"]
        or Path(cfg["output_root"]) != args.root
        or len(complete["rows"]) != 6
        or type(complete["actual_causal_rollout_steps"]) is not int
        or complete["actual_causal_rollout_steps"] != 18000
        or complete["reused_original_controls"] != 6
        or any(
            complete[k] is not False
            for k in (
                "new_neural_policy_trained",
                "independent_physics_qualified",
                "teacher_qualified",
                "hardware_authorized",
                "promotion_authorized",
            )
        )
    ):
        raise ValueError("six complete unqualified development rollouts required")
    for name, h in declaration["input_file_hashes"].items():
        pin(name, h)
    stage = sealed(cfg["stage_readback"])
    native = sealed(stage["effective_cohort_spec"]["producer_complete"])
    original_declaration = sealed(stage["effective_cohort_spec"]["producer_declaration"])
    if len(native["results"]) != 6:
        raise ValueError("all six original controls required")
    rows, total_forecast = [], 0
    for i, (row, parent) in enumerate(zip(complete["rows"], native["results"], strict=True)):
        if (
            sealed(args.root / f"case-{i}-complete.json") != row
            or type(row["index"]) is not int
            or row["index"] != i
            or row["source_receipt_hash"] != parent["receipt_hash"]
            or row["job"] != original_declaration["scope"]["jobs"][i]
            or row["original_outcome"] != parent["outcome"]
        ):
            raise ValueError("original ordered cases including failures required")
        trace = arrays(row["arrays"])
        original = arrays(parent["witnesses"]["trace"])
        extra = {
            "early_impact_feedback_" + s
            for s in ("requested_offset_nm", "applied_offset_nm", "forecast_steps")
        }
        if set(trace) != set(original) | extra or any(
            len(a) != 300 or a.dtype.kind not in "fiub" or not np.isfinite(a).all()
            for a in trace.values()
        ):
            raise ValueError(
                "all original fields and complete finite300-frame early captures required"
            )
        verify_actual_torques(trace)
        requested, applied = intervention_frames(trace)
        if (
            requested != row["first_requested_correction_frame"]
            or applied != row["first_physically_applied_correction_frame"]
        ):
            raise ValueError("actual torque-intervention boundaries differ")
        require_prefix(trace, original, applied)
        outcome = label_first_touch(trace).to_dict()
        if outcome != row["causal_rollout_outcome"]:
            raise ValueError("raw full original M0 outcome differs")
        review = row["causal_review"]
        forecast_steps = trace["early_impact_feedback_forecast_steps"]
        if (
            forecast_steps.shape != (300, 1, 1)
            or forecast_steps.dtype != np.int64
            or np.any(forecast_steps < 0)
            or np.any(forecast_steps % 40)
        ):
            raise ValueError("whole typed per-frame physical forecast counts required")
        forecast_count = int(np.sum(forecast_steps))
        if (
            review["native_integrated_steps"] != 3000
            or review["native_attempted_steps"] != 3000
            or review["forecast_integrated_steps"] != forecast_count
            or review["forecast_attempted_steps"] != forecast_count
            or not 0 <= forecast_count <= 1800
            or forecast_count % 40
            or review["source_future_trace_or_targets_consumed"] is not False
        ):
            raise ValueError("whole body and forecast budgets/current-state boundary required")
        selections = read_decisions(
            trace,
            args.root / f"case-{i}-preaction-evidence",
            sealed=sealed,
            arrays=arrays,
        )
        total_forecast += forecast_count
        rows.append(
            dict(
                index=i,
                job=row["job"],
                original_outcome=parent["outcome"],
                actual_outcome=outcome,
                selected_window_alignment=selections,
                forecast_steps=forecast_count,
            )
        )
    before, after = [r["original_outcome"] for r in rows], [r["actual_outcome"] for r in rows]
    totals = dict(
        actual_forecast_steps=total_forecast,
        original_passes=sum(r["task_pass"] for r in before),
        candidate_passes=sum(r["task_pass"] for r in after),
        rescues=sum(
            not a["task_pass"] and b["task_pass"] for a, b in zip(before, after, strict=True)
        ),
        regressions=sum(
            a["task_pass"] and not b["task_pass"] for a, b in zip(before, after, strict=True)
        ),
        added_unsafe=sum(a["safe"] and not b["safe"] for a, b in zip(before, after, strict=True)),
        new_optimizer_updates=0,
    )
    if (
        any(type(complete[k]) is not int or complete[k] != v for k, v in totals.items())
        or total_forecast > 10800
    ):
        raise ValueError("all primitive task and execution totals differ")
    if any(digest(name) != h for name, h in pins.items()):
        raise ValueError("source/input changed during whole readback")
    report = dict(
        schema="soccer.rsi.early_impact_h300_readback.v1",
        source_head=args.source_head,
        complete_hash=complete["report_hash"],
        input_file_hashes=pins,
        rows=rows,
        **totals,
        all_six_including_failures_retained=True,
        new_physical_steps=0,
        new_neural_inferences=0,
        shared_label_and_score_algorithms=True,
        independent_neural_implementation=False,
        teacher_qualified=False,
        hardware_authorized=False,
        promotion_authorized=False,
    )
    report["report_hash"] = hash_json(report)
    write_once(args.output, report)
    print("WHOLE_EARLY_H300_READBACK", report["report_hash"], flush=True)


if __name__ == "__main__":
    main()
