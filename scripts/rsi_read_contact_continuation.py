"""Whole-six continuation bias readback, with original native/dense clocks."""

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np

from rosclaw_soccer.sim.contracts import hash_json
from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_calibrate_contact_continuation import model_bias, require_scope
from scripts.rsi_impact_torque_authority_screen import equal
from scripts.rsi_stage_release_contact_replay import digest, require_terminal


def verify_arrays(held, repeated, oracle, trace, geometry):
    names = {
        "canonical_qpos_after_substep",
        "canonical_qvel_after_substep",
        "requested_pd_torque_nm",
        "applied_control_nm",
        "actual_actuator_force_nm",
        "pelvis_height_solver_epoch_m",
        "robot_force_on_ball_world_n",
        "completed_foot_ground_normal_n",
    }
    for values in (held, repeated, oracle):
        if set(values) != names:
            raise ValueError("all eight ordinary raw prediction arrays required")
        if any(
            a.dtype != np.float64 or a.shape[0] != 20 or not np.isfinite(a).all()
            for a in values.values()
        ):
            raise ValueError("whole finite20-substep arrays required")
    for name in names:
        a, b, o = held[name], repeated[name], oracle[name]
        if a.shape != b.shape or a.tobytes() != b.tobytes() or a[:10].tobytes() != o[:10].tobytes():
            raise ValueError("all repeated bytes and original20ms prefix required")
    contact = geometry["contact"]
    frame, offset = divmod(contact["first_foot_substep"], 10)
    for i in range(2):
        equal(
            oracle["canonical_qpos_after_substep"][i * 10 + 9],
            trace["canonical_qpos"][frame + i + 1, 0],
            "original40ms frame qpos",
        )
        equal(
            oracle["canonical_qvel_after_substep"][i * 10 + 9],
            trace["canonical_qvel"][frame + i + 1, 0],
            "original40ms frame qvel",
        )
    for predicted, original, width in (
        ("requested_pd_torque_nm", "pd_requested_torque_nm", 29),
        ("applied_control_nm", "torque_nm", 29),
        ("actual_actuator_force_nm", "actual_actuator_force_nm", 29),
        ("pelvis_height_solver_epoch_m", "pelvis_z_per_substep_m", None),
    ):
        shape = (20,) if width is None else (20, width)
        equal(
            oracle[predicted],
            trace[original][frame : frame + 2, 0].reshape(shape),
            "all original40ms " + predicted,
        )
    equal(
        oracle["robot_force_on_ball_world_n"][offset : offset + 10].sum(axis=0) * 0.002,
        contact["robot_impulse_initial_window_world_n_s"],
        "original dense first10 contact impulse",
    )
    equal(
        oracle["canonical_qvel_after_substep"][offset + 9, 35:38],
        contact["ball_velocity_after_initial_window_world_m_s"],
        "original dense exact initial-window velocity",
    )
    return model_bias(held, oracle, first_contact_substep=offset)


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
        raise ValueError("unused readback, exact frozen source and complete evidence required")
    require_terminal(args.unit)
    pins = {
        str(p): digest(p)
        for folder in (source / "src", source / "scripts")
        for p in folder.rglob("*.py")
    }

    def sealed(path, key="report_hash"):
        if path.is_symlink():
            raise ValueError("ordinary whole source artifact required")
        h = digest(path)
        if str(path) in pins and pins[str(path)] != h:
            raise ValueError("source evidence changed")
        d = json.loads(path.read_text())
        if d[key] != hash_json({k: v for k, v in d.items() if k != key}):
            raise ValueError("whole artifact logical seal differs")
        pins[str(path)] = h
        return d

    complete, declaration = (sealed(args.root / n) for n in ("complete.json", "declaration.json"))
    cfg = declaration["config"]
    require_scope(cfg)
    for name, h in declaration["input_file_hashes"].items():
        if (name in pins and pins[name] != h) or digest(name) != h:
            raise ValueError("whole original model/source/native/dense input changed")
        pins[name] = h
    if (
        complete["schema"] != "soccer.rsi.contact_continuation_calibration_complete.v1"
        or complete["declaration_hash"] != declaration["report_hash"]
        or complete["all_six_source_cases_retained"] is not True
        or len(complete["rows"]) != 6
        or any(
            type(complete[k]) is not int or complete[k] != n
            for k, n in (
                ("recorded_attempted_steps", 18000),
                ("recorded_integrated_steps", 18000),
                ("held_attempted_steps", 240),
                ("held_integrated_steps", 240),
                ("oracle_attempted_steps", 120),
                ("oracle_integrated_steps", 120),
                ("combined_attempted_steps", 18360),
                ("combined_integrated_steps", 18360),
                ("new_closed_loop_policy_trials", 0),
                ("new_neural_inferences", 0),
                ("new_optimizer_updates", 0),
                ("candidate_actions_optimized", 0),
            )
        )
        or any(
            complete[k] is not False
            for k in ("teacher_qualified", "hardware_authorized", "promotion_authorized")
        )
    ):
        raise ValueError("whole six original bounded calibration required")
    for name in (cfg["geometry_readback"], cfg["stage_readback"]):
        if str(Path(name)) not in pins:
            raise ValueError("original geometry/stage must be in producer authentication chain")
    geometry, stage = sealed(Path(cfg["geometry_readback"])), sealed(Path(cfg["stage_readback"]))
    spec = stage["effective_cohort_spec"]
    native = sealed(Path(spec["producer_complete"]))
    if (
        native["report_hash"] != declaration["native_complete_hash"]
        or geometry["report_hash"] != declaration["original_geometry_hash"]
    ):
        raise ValueError("same original six native bodies and dense geometry required")
    rows = []
    for index, (row, parent, geom) in enumerate(
        zip(complete["rows"], native["results"], geometry["rows"], strict=True)
    ):
        if (
            sealed(args.root / f"case-{index}-complete.json") != row
            or row["index"] != index
            or row["job"] != geom["job"]
            or row["source_receipt_hash"] != parent["receipt_hash"]
            or row["original_outcome"] != parent["outcome"]
            or any(
                row[k] is not True
                for k in (
                    "eligible",
                    "repeat_held_prediction_bit_equal",
                    "first20ms_held_prediction_matches_original",
                    "oracle40ms_matches_original_every_substep",
                    "future_targets_not_input_to_held_predictor",
                )
            )
        ):
            raise ValueError("all original ordered cases and source parity required")
        views = {}
        for name in ("held", "repeated", "oracle"):
            expected = args.root / f"case-{index}-{name}.npz"
            w = row["artifacts"][name]
            if w["path"] != str(expected) or digest(expected) != w["sha256"]:
                raise ValueError("every complete raw prediction required")
            pins[str(expected)] = w["sha256"]
            with np.load(expected, allow_pickle=False) as archive:
                views[name] = dict(archive)
        trace_path = parent["witnesses"]["trace"]["path"]
        if pins.get(trace_path) != parent["witnesses"]["trace"]["sha256"]:
            raise ValueError("original native trace must already be authenticated")
        with np.load(trace_path, allow_pickle=False) as archive:
            trace = dict(archive)
        bias = verify_arrays(views["held"], views["repeated"], views["oracle"], trace, geom)
        if bias != row["bias"]:
            raise ValueError("bias metrics not independently reproduced")
        rows.append(
            dict(index=index, job=row["job"], original_outcome=row["original_outcome"], bias=bias)
        )
    admissible = all(r["bias"]["frozen_limits_satisfied"] for r in rows)
    if admissible is not complete["held_target_prediction_within_frozen_bias_limits_all_six"]:
        raise ValueError("whole-cohort model bias decision differs")
    if any(digest(path) != h for path, h in pins.items()):
        raise ValueError("original source/input changed during readback")
    report = dict(
        schema="soccer.rsi.contact_continuation_calibration_readback.v1",
        source_head=args.source_head,
        complete_hash=complete["report_hash"],
        input_file_hashes=pins,
        rows=rows,
        all_six_raw_repeat_arrays_bit_equal=True,
        all_six_original20ms_prefixes_bit_equal=True,
        all_six_oracles_match_native_and_dense_contact_clocks=True,
        held_target_within_frozen_bias_limits_all_six=admissible,
        whole_cohort_held_target_teacher_admissible=False,
        new_physical_steps=0,
        new_neural_inferences=0,
        new_optimizer_updates=0,
        teacher_qualified=False,
        hardware_authorized=False,
        promotion_authorized=False,
    )
    report["report_hash"] = hash_json(report)
    write_once(args.output, report)
    print(
        "CONTINUATION_WHOLE_READBACK",
        report["report_hash"],
        "WITHIN_ALL_LIMITS",
        admissible,
        flush=True,
    )


if __name__ == "__main__":
    main()
