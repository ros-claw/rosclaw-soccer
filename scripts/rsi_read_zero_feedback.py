"""Independently compare every zero-feedback substep with original dense physics."""

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np

from rosclaw_soccer.sim.contracts import hash_json
from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_feedback_branch_inputs import require_scope
from scripts.rsi_impact_torque_authority_screen import equal
from scripts.rsi_stage_release_contact_replay import digest, require_terminal


def verify_branch(a, trace, dense, frame):
    end = a["end_integration_state"]
    if (
        end.ndim != 1
        or end.dtype != np.float64
        or not 41 <= len(end) <= 65536
        or not np.isfinite(end).all()
    ):
        raise ValueError("whole finite recorded end integration state required")
    for key, shape in (
        ("canonical_qpos_after_substep", (20, 43)),
        ("canonical_qvel_after_substep", (20, 41)),
        ("requested_pd_torque_nm", (20, 29)),
        ("applied_control_nm", (20, 29)),
        ("actual_actuator_force_nm", (20, 29)),
        ("pelvis_height_solver_epoch_m", (20,)),
        ("robot_force_on_ball_world_n", (20, 3)),
        ("first_control_peak_force_n", (6,)),
    ):
        v = a[key]
        if v.shape != shape or v.dtype != np.float64 or not np.isfinite(v).all():
            raise ValueError("whole finite40ms physical branch arrays required")
    start = frame * 10
    equal(
        a["canonical_qpos_after_substep"],
        dense["canonical_qpos_before_substep"][start + 1 : start + 21],
        "all20 original dense qpos",
    )
    equal(
        a["canonical_qvel_after_substep"],
        dense["canonical_qvel_after_substep"][start : start + 20],
        "all20 original dense qvel",
    )
    equal(
        a["robot_force_on_ball_world_n"],
        dense["robot_contact_force_on_ball_world_n"][start : start + 20],
        "all20 original dense true force",
    )
    for predicted, original, shape in (
        ("requested_pd_torque_nm", "pd_requested_torque_nm", (20, 29)),
        ("applied_control_nm", "torque_nm", (20, 29)),
        ("actual_actuator_force_nm", "actual_actuator_force_nm", (20, 29)),
        ("pelvis_height_solver_epoch_m", "pelvis_z_per_substep_m", (20,)),
    ):
        equal(
            a[predicted],
            trace[original][frame : frame + 2, 0].reshape(shape),
            "source native " + predicted,
        )
    equal(a["first_control_peak_force_n"], trace["force_n"][frame, 0], "original causal force bin")
    for name, original in (
        ("navigation_target_rad", "foundation_navigation_joint_target_rad"),
        ("taskspace_increment_rad", "taskspace_joint_increment_rad"),
        ("motor_delta_rad", "motor_delta_rad"),
        ("joint_target_rad", "joint_target_rad"),
        ("canonical_qpos", "canonical_qpos"),
        ("canonical_qvel", "canonical_qvel"),
    ):
        equal(
            a["next_" + name],
            trace[original][frame + 1, 0],
            "original full next target/body " + name,
        )
    for name, width in (
        ("encoder_features", 640),
        ("latent_token", 64),
        ("decoder_input", 994),
        ("raw_action_isaac", 29),
        ("joint_target_mujoco_rad", 29),
    ):
        value = a["neural_next_" + name]
        if (
            value.shape != (1, width)
            or value.dtype != np.float32
            or not np.isfinite(value).all()
            or value.tobytes() != trace["foundation_neural_" + name][frame + 1].tobytes()
        ):
            raise ValueError("actual next-frame neural input/output bits differ")


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
        subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
        != args.source_head
        or subprocess.check_output(["git", "status", "--porcelain"], cwd=source, text=True).strip()
        or args.output.exists()
        or (args.root / "failure.json").exists()
        or digest(args.root / "complete.json") != args.complete_file_hash
    ):
        raise ValueError(
            "exact clean frozen source, original complete bytes and unused readback required"
        )
    require_terminal(args.unit)
    pins = {
        str(p): digest(p)
        for folder in (source / "src", source / "scripts")
        for p in folder.rglob("*.py")
    }

    def sealed(path, key="report_hash"):
        if path.is_symlink():
            raise ValueError("ordinary original artifact required")
        h = digest(path)
        if str(path) in pins and pins[str(path)] != h:
            raise ValueError("original input changed")
        value = json.loads(path.read_text())
        if value[key] != hash_json({k: v for k, v in value.items() if k != key}):
            raise ValueError("whole original artifact seal differs")
        pins[str(path)] = h
        return value

    complete, declaration = (sealed(args.root / n) for n in ("complete.json", "declaration.json"))
    cfg = declaration["config"]
    require_scope(cfg)
    for name, h in declaration["input_file_hashes"].items():
        if (name in pins and pins[name] != h) or digest(name) != h:
            raise ValueError("complete original source/model/body/dense proof changed")
        pins[name] = h
    if (
        complete["schema"] != "soccer.rsi.zero_feedback_calibration_complete.v1"
        or complete["declaration_hash"] != declaration["report_hash"]
        or len(complete["rows"]) != 6
        or any(
            complete[k] is not True
            for k in ("all_six_original_cases_retained", "all_six_zero_feedback_branches_qualified")
        )
        or any(
            type(complete[k]) is not int or complete[k] != n
            for k, n in (
                ("recorded_attempted_steps", 18000),
                ("recorded_integrated_steps", 18000),
                ("forecast_attempted_steps", 240),
                ("forecast_integrated_steps", 240),
                ("combined_attempted_steps", 18240),
                ("combined_integrated_steps", 18240),
                ("planner_runs_attempted", 60),
                ("current_pairs_completed", 6),
                ("original_residual_decisions_completed", 1800),
                ("original_recurrent_updates_completed", 1620),
                ("branch_target_attempts", 12),
                ("branch_target_completed", 12),
                ("new_closed_loop_policy_trials", 0),
                ("new_optimizer_updates", 0),
                ("candidate_actions_optimized", 0),
            )
        )
        or any(
            complete[k] is not False
            for k in ("teacher_qualified", "hardware_authorized", "promotion_authorized")
        )
    ):
        raise ValueError("whole original bounded zero-feedback result required")
    stage = sealed(Path(cfg["stage_readback"]))
    native = sealed(Path(stage["effective_cohort_spec"]["producer_complete"]))
    native_declaration = sealed(Path(stage["effective_cohort_spec"]["producer_declaration"]))
    dense = sealed(Path(stage["effective_contact_spec"]["replay_complete"]))
    if (
        native["report_hash"] != declaration["native_complete_hash"]
        or len(native["results"]) != 6
        or len(dense["rows"]) != 6
    ):
        raise ValueError("same whole original native and independent dense physics required")
    rows = []
    for index, (row, parent, contact) in enumerate(
        zip(complete["rows"], native["results"], dense["rows"], strict=True)
    ):
        if (
            sealed(args.root / f"case-{index}-complete.json") != row
            or row["index"] != index
            or row["source_receipt_hash"] != parent["receipt_hash"]
            or row["source_receipt_hash"] != contact["source_receipt_hash"]
            or row["original_outcome"] != parent["outcome"]
            or row["job"] != native_declaration["scope"]["jobs"][index]
            or len(row["predictions"]) != 2
        ):
            raise ValueError("every original ordered case including failures required")
        frame = row["retrospective_frame"]
        if (
            type(frame) is not int
            or frame != parent["outcome"]["first_foot_contact_frame"]
            or not 30 <= frame < 298
        ):
            raise ValueError("unchanged original diagnostic frame required")

        def load_arrays(w):
            if str(w["path"]) not in pins:
                pins[w["path"]] = w["sha256"]
            if pins[w["path"]] != w["sha256"] or digest(w["path"]) != w["sha256"]:
                raise ValueError("whole ordinary raw original array bytes required")
            with np.load(w["path"], allow_pickle=False) as a:
                return dict(a)

        trace = load_arrays(parent["witnesses"]["trace"])
        dense_arrays = load_arrays(contact["arrays"])
        branches = []
        for number, w in enumerate(row["predictions"]):
            if w["path"] != str(args.root / f"case-{index}-branch-{number}.npz"):
                raise ValueError("same unused original branch paths required")
            arrays = load_arrays(w)
            verify_branch(arrays, trace, dense_arrays, frame)
            branches.append(arrays)
        if (
            set(branches[0]) != set(branches[1])
            or any(
                branches[0][k].dtype != branches[1][k].dtype
                or branches[0][k].shape != branches[1][k].shape
                or branches[0][k].tobytes() != branches[1][k].tobytes()
                for k in branches[0]
            )
            or row["predictions"][0]["donor_hash"] != row["predictions"][1]["donor_hash"]
        ):
            raise ValueError("all independent zero-branch bytes must match")
        rows.append(
            dict(
                index=index,
                job=row["job"],
                original_outcome=row["original_outcome"],
                source_receipt_hash=row["source_receipt_hash"],
                all20_substeps_match_original_dense=True,
                all_neural_next_capture_bits_match=True,
                next_target_decomposition_matches=True,
                repeated_whole_branch_bytes_equal=True,
            )
        )
    if any(digest(path) != h for path, h in pins.items()):
        raise ValueError("original source/input changed during readback")
    report = dict(
        schema="soccer.rsi.zero_feedback_calibration_readback.v1",
        source_head=args.source_head,
        complete_hash=complete["report_hash"],
        input_file_hashes=pins,
        rows=rows,
        all_six_no_intervention_feedback_branches_qualified=True,
        nonzero_interventions_qualified=False,
        full_h300_teacher_qualified=False,
        new_physical_steps=0,
        new_optimizer_updates=0,
        teacher_qualified=False,
        hardware_authorized=False,
        promotion_authorized=False,
    )
    report["report_hash"] = hash_json(report)
    write_once(args.output, report)
    print("WHOLE_ZERO_FEEDBACK_READBACK", report["report_hash"], flush=True)


if __name__ == "__main__":
    main()
