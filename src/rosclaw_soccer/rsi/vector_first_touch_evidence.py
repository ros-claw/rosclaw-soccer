"""Fail-closed audit of independent SIM_ONLY Isaac first-touch environments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.rsi.first_touch_course_catalog import (
    sample_training_courses,
    static_development_courses,
)
from rosclaw_soccer.rsi.physical_report_io import load_physical_report
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json


def audit_body_trace(folder: Path, report: dict[str, Any], frames: int, lanes: int) -> int | None:
    """Verify optional proprioceptive sequence and return changed command frames."""
    if "body_trace_hash" not in report:
        return None
    body_trace_path = folder / "body_trace.npz"
    if not body_trace_path.is_file() or report["body_trace_hash"] != hash_bytes(
        body_trace_path.read_bytes()
    ):
        raise ValueError("unauthenticated G1 body trace")
    with np.load(body_trace_path, allow_pickle=False) as body:
        shapes = {
            "root_pose_xyzw_m": (frames, lanes, 7),
            "root_velocity_world": (frames, lanes, 6),
            "joint_position_rad": (frames, lanes, 29),
            "joint_velocity_rad_s": (frames, lanes, 29),
            "joint_target_rad": (frames, lanes, 29),
            "navigation_speed_mps": (frames, lanes),
        }
        if "navigation_lateral_ball_gain" in report:
            if report["navigation_lateral_ball_gain"] not in (0.0, 0.8, 1.0, 1.2):
                raise ValueError("unbounded lateral approach gain")
            shapes["navigation_lateral_speed_mps"] = (frames, lanes)
        temporal = report.get("schema") == "rsi_isaac_vector_first_touch_temporal_candidate_v1"
        foot_geometry = "foot_geometry_body_names" in report
        if foot_geometry:
            if report["foot_geometry_body_names"] != [
                "left_ankle_roll_link",
                "right_ankle_roll_link",
                "left_knee_link",
                "right_knee_link",
            ]:
                raise ValueError("uncommitted foot geometry body order")
            shapes.update(
                ball_position_before_step_m=(frames, lanes, 3),
                ball_linear_velocity_before_step_m_s=(frames, lanes, 3),
                foot_geometry_position_before_step_m=(frames, lanes, 4, 3),
                foot_geometry_velocity_before_step_m_s=(frames, lanes, 4, 3),
            )
        if temporal:
            shapes.update(
                ball_position_before_step_m=(frames, lanes, 3),
                ball_linear_velocity_before_step_m_s=(frames, lanes, 3),
                baseline_joint_target_rad=(frames, lanes, 29),
                applied_residual_rad=(frames, lanes, 3),
            )
        if set(body.files) != set(shapes) or any(
            body[key].shape != shape or not np.isfinite(body[key]).all()
            for key, shape in shapes.items()
        ):
            raise ValueError("invalid G1 body trace shape or finite values")
        if foot_geometry:
            with np.load(folder / "trace.npz", allow_pickle=False) as physics:
                if not np.allclose(
                    body["ball_position_before_step_m"][1:],
                    physics["ball_position_m"][:-1],
                    atol=1e-5,
                    rtol=0,
                ):
                    raise ValueError("foot geometry ball alignment invalid")
        command = body["navigation_speed_mps"]
        base_speed = report.get("navigation_speed_mps", 1.4)
        allowed = [base_speed]
        if "near_ball_speed_mps" in report:
            allowed.append(report["near_ball_speed_mps"])
        if not np.isin(command, allowed).all():
            raise ValueError("G1 body trace has uncommitted navigation speed")
        if temporal:
            from rosclaw_soccer.providers.g1.joint_contract import G1_DDS_JOINT_NAMES
            from rosclaw_soccer.rsi.temporal_first_touch_policy import JOINT_NAMES

            joint_indices = [G1_DDS_JOINT_NAMES.index(name) for name in JOINT_NAMES]
            actual_difference = body["joint_target_rad"] - body["baseline_joint_target_rad"]
            residual = body["applied_residual_rad"]
            remaining = np.delete(actual_difference, joint_indices, axis=2)
            if (
                np.max(np.abs(residual)) > 0.08 + 1e-6
                or not np.allclose(actual_difference[:, :, joint_indices], residual, atol=1e-5)
                or not np.allclose(remaining, 0.0, atol=1e-5)
            ):
                raise ValueError("temporal body trace violates bounded joint target delta")
        return int(np.count_nonzero(command != base_speed))


def audit_vector_first_touch(
    folder: Path, *, decoder_sink: list[Any] | None = None
) -> dict[str, Any]:
    report = load_physical_report(folder / "report.json")
    trace_path = folder / "trace.npz"
    committed = {key: value for key, value in report.items() if key != "report_hash"}
    if (
        report.get("schema")
        not in (
            "rsi_isaac_vector_first_touch_smoke_v1",
            "rsi_isaac_vector_first_touch_late_swing_v1",
        )
        or report.get("activation_ceiling") != "SIM_ONLY"
        or report.get("learning_authorized") is not False
        or report.get("promotion_authorized") is not False
        or report.get("report_hash") != hash_json(committed)
        or report.get("trace_hash") != hash_bytes(trace_path.read_bytes())
        or type(report.get("navigation_speed_mps", 1.4)) not in (int, float)
        or not np.isfinite(report.get("navigation_speed_mps", 1.4))
        or not 0.8 <= report.get("navigation_speed_mps", 1.4) <= 1.5
        or (
            "planner_seed" in report
            and (
                type(report["planner_seed"]) is not int
                or not 0 <= report["planner_seed"] <= 2**31 - 3000
                or report["planner_seed"] == 30300
            )
        )
        or (("near_ball_gap_m" in report) != ("near_ball_speed_mps" in report))
        or (("near_ball_gap_m" in report) != ("near_ball_incoming_only" in report))
        or (
            "near_ball_gap_m" in report
            and (
                type(report["near_ball_gap_m"]) not in (int, float)
                or not np.isfinite(report["near_ball_gap_m"])
                or not 0.6 <= report["near_ball_gap_m"] <= 1.6
                or type(report["near_ball_speed_mps"]) not in (int, float)
                or not np.isfinite(report["near_ball_speed_mps"])
                or not 0.8 <= report["near_ball_speed_mps"] <= 1.5
                or type(report["near_ball_incoming_only"]) is not bool
            )
        )
        or (
            "onnx_graph_encoder_layout" in report
            and report["onnx_graph_encoder_layout"] is not True
        )
        or (
            "torch_batch_shadow" in report
            and (
                report["torch_batch_shadow"] is not True
                or type(report.get("torch_batch_drive")) is not bool
                or type(report.get("torch_batch_max_target_difference_rad")) not in (int, float)
                or not np.isfinite(report["torch_batch_max_target_difference_rad"])
                or not 0 <= report["torch_batch_max_target_difference_rad"] <= 1e-3
            )
        )
        or (
            "torch_batch_shadow" not in report
            and ("torch_batch_drive" in report or "torch_batch_max_target_difference_rad" in report)
        )
        or (
            "torch_batch_plan_only" in report
            and (
                report["torch_batch_plan_only"] is not True
                or "torch_batch_shadow" in report
                or type(report.get("torch_batch_max_internal_target_difference_rad"))
                not in (int, float)
                or not np.isfinite(report["torch_batch_max_internal_target_difference_rad"])
                or not 0 <= report["torch_batch_max_internal_target_difference_rad"] <= 1e-3
            )
        )
        or (
            "torch_batch_plan_only" not in report
            and "torch_batch_max_internal_target_difference_rad" in report
        )
    ):
        raise ValueError("unauthenticated vector first-touch evidence")
    entries = report.get("environments")
    frames = report.get("frames")
    single_course_lane = report.get("single_course_lane")
    static_single_lane = report.get("static_single_course_lane")
    if (
        not isinstance(entries, list)
        or not 1 <= len(entries) <= 16
        or (len(entries) == 1) != (single_course_lane is not None or static_single_lane is not None)
        or (single_course_lane is not None and static_single_lane is not None)
        or (
            single_course_lane is not None
            and (type(single_course_lane) is not int or not 0 <= single_course_lane < 16)
        )
        or type(frames) is not int
        or not 50 <= frames <= 400
        or [row.get("environment") for row in entries] != list(range(len(entries)))
    ):
        raise ValueError("invalid independent environment ledger")
    with np.load(trace_path, allow_pickle=False) as archive:
        if set(archive.files) != {
            "ball_position_m",
            "ball_angular_velocity_rad_s",
            "ball_body_contact_force_peak_n",
        }:
            raise ValueError("vector trace contract changed")
        positions = archive["ball_position_m"]
        spin = archive["ball_angular_velocity_rad_s"]
        force = archive["ball_body_contact_force_peak_n"]
    n = len(entries)
    if (
        positions.shape != (frames, n, 3)
        or spin.shape != (frames, n, 3)
        or force.shape != (frames, n, 6)
        or not np.isfinite(positions).all()
        or not np.isfinite(spin).all()
        or not np.isfinite(force).all()
        or np.any(force < 0)
    ):
        raise ValueError("invalid vector physics trace")
    changed_command_frames = audit_body_trace(folder, report, frames, n)
    lanes = np.asarray([row["lane_y_m"] for row in entries], dtype=np.float64)
    lane_spacing = report.get("lane_spacing_m", 8.0)
    if (
        lane_spacing not in (8.0, 24.0)
        or (lane_spacing == 24.0 and n != 16)
        or not np.isfinite(lanes).all()
        or not np.allclose(lanes, np.arange(n) * lane_spacing, atol=1e-6, rtol=0)
        or (n > 1 and np.min(np.diff(lanes)) < 6.0)
    ):
        raise ValueError("training lanes are not physically isolated")
    excursion = float(np.max(np.abs(positions[:, :, 1] - lanes[None, :])))
    max_isolated_excursion = (
        10.0
        if lane_spacing == 24.0
        else 6.0
        if report.get("schema") == "rsi_isaac_vector_first_touch_late_swing_v1"
        else 4.0
    )
    if n == 1 and "single_instance_max_lateral_excursion_m" in report:
        if not np.isclose(report["single_instance_max_lateral_excursion_m"], excursion):
            raise ValueError("single-instance ball excursion disagrees with physical trace")
    elif n == 1 and excursion >= 4.0:
        raise ValueError("unauthenticated single-instance ball excursion")
    if report.get("schema") == "rsi_isaac_vector_first_touch_late_swing_v1":
        if n == 1:
            if (
                report.get("minimum_cross_robot_distance_m") is not None
                or report.get("minimum_cross_ball_distance_m") is not None
            ):
                raise ValueError("single-instance isolation contract changed")
        else:
            with np.load(folder / "body_trace.npz", allow_pickle=False) as body:
                roots = body["root_pose_xyzw_m"][:, :, :3]
            robot_distance = np.linalg.norm(
                positions[:, :, None, :] - roots[:, None, :, :], axis=-1
            )
            ball_distance = np.linalg.norm(
                positions[:, :, None, :] - positions[:, None, :, :], axis=-1
            )
            diagonal = np.eye(n, dtype=np.bool_)
            robot_distance[:, diagonal] = np.inf
            ball_distance[:, diagonal] = np.inf
            min_robot = float(np.min(robot_distance))
            min_ball = float(np.min(ball_distance))
            if (
                excursion >= max_isolated_excursion
                or min_robot <= 2.0
                or min_ball <= 1.0
                or not np.isclose(report.get("minimum_cross_robot_distance_m", np.nan), min_robot)
                or not np.isclose(report.get("minimum_cross_ball_distance_m", np.nan), min_ball)
            ):
                raise ValueError("late-swing lane isolation not physically verified")
    elif n > 1 and excursion >= max_isolated_excursion:
        raise ValueError("ball escaped its lane")
    courses = []
    training_seed = report.get("training_course_seed")
    if training_seed is not None:
        if (
            type(training_seed) is not int
            or n not in (1, 16)
            or "course_catalog_hash" not in report
            or report["course_catalog_hash"]
            != hash_json(sample_training_courses(training_seed, 16))
        ):
            raise ValueError("training course split is unauthenticated")
    elif "course_catalog_hash" in report:
        raise ValueError("unbound course catalog hash")
    clean = 0
    positives = 0
    sides: set[str] = set()
    for i, row in enumerate(entries):
        course = row["course"]
        key = (course["ball_x_m"], course["ball_y_local_m"], course["ball_vx_m_s"])
        courses.append(key)
        if (
            (
                training_seed is None
                and (
                    key[0] not in ((2.3, 2.4, 2.6, 2.7) if n == 16 else (2.4, 2.6))
                    or key[1] not in (-0.1, 0.1)
                    or key[2] not in (-0.5, 0.5)
                )
            )
            or not 0.65 <= row["minimum_pelvis_z_m"] <= 1.5
            or not np.allclose(
                row["ball_final_local_xyz_m"],
                (positions[-1, i, 0], positions[-1, i, 1] - lanes[i], positions[-1, i, 2]),
                atol=1e-5,
                rtol=0,
            )
        ):
            raise ValueError("course, body safety or final ball position disagrees")
        active = np.flatnonzero(np.max(force[:, i], axis=1) > 1.0)
        first = int(active[0]) if len(active) else None
        bodies = np.flatnonzero(np.max(force[:, i], axis=0) > 1.0).tolist()
        if row["first_contact_frame"] != first or row["contact_body_indices"] != bodies:
            raise ValueError("contact ledger disagrees with authenticated trace")
        positives += first is not None
        clean_foot_only = bool(bodies and set(bodies) <= {0, 1})
        clean += clean_foot_only
        if clean_foot_only and first is not None:
            sides.add("left" if force[first, i, 0] >= force[first, i, 1] else "right")
    if len(set(courses)) != len(courses):
        raise ValueError("duplicate courses are not independent episodes")
    expected_courses = (
        (sample_training_courses(training_seed, 16)[single_course_lane],)
        if training_seed is not None and single_course_lane is not None
        else sample_training_courses(training_seed, 16)
        if training_seed is not None
        else None
    )
    if training_seed is not None and tuple(courses) != expected_courses:
        raise ValueError("training courses differ from committed seed")
    if static_single_lane is not None and (
        type(static_single_lane) is not int
        or static_single_lane not in (0, 1)
        or n != 1
        or training_seed is not None
        or tuple(courses) != (static_development_courses(2)[static_single_lane],)
    ):
        raise ValueError("static single course differs from committed lane")
    result = {
        "schema": "rsi_isaac_vector_first_touch_audit_v1",
        "activation_ceiling": "SIM_ONLY",
        "source_report_hash": report["report_hash"],
        "independent_physical_episode_count": n,
        "any_contact_episode_count": positives,
        "clean_foot_only_episode_count": clean,
        "positive_foot_sides": sorted(sides),
        "imitation_training_authorized": bool(n >= 8 and clean >= 4 and sides == {"left", "right"}),
        "promotion_authorized": False,
    }
    if changed_command_frames is not None:
        result["recorded_body_frames"] = frames
        result["changed_navigation_command_frames"] = changed_command_frames
    motor_keys = {"contact_motor_policy", "contact_motor_policy_hash", "contact_motor_trace_hash"}
    if motor_keys & report.keys():
        if not motor_keys <= report.keys():
            raise ValueError("incomplete motor policy evidence")
        from rosclaw_soccer.rsi.contact_motor_evidence import audit_motor_execution

        if decoder_sink is None:
            result.update(audit_motor_execution(folder, report))
        else:
            result.update(audit_motor_execution(folder, report, decoder_sink=decoder_sink))
    elif decoder_sink is not None:
        raise ValueError("motor evidence required for decoder capture")
    result["report_hash"] = hash_json(result)
    return result


def audit_reset_replay(folder: Path) -> dict[str, Any]:
    """Verify paired in-process reset without counting replay as new courses."""

    baseline = audit_vector_first_touch(folder)
    source = load_physical_report(folder / "report.json")
    replay_report = json.loads((folder / "reset_report.json").read_text(encoding="utf-8"))
    replay_trace = folder / "reset_replay.npz"
    committed = {key: value for key, value in replay_report.items() if key != "report_hash"}
    if (
        replay_report.get("schema") != "rsi_isaac_vector_first_touch_reset_replay_v1"
        or replay_report.get("activation_ceiling") != "SIM_ONLY"
        or replay_report.get("learning_authorized") is not False
        or replay_report.get("promotion_authorized") is not False
        or replay_report.get("source_report_hash") != source["report_hash"]
        or replay_report.get("source_hash") != source["source_hash"]
        or replay_report.get("replay_trace_hash") != hash_bytes(replay_trace.read_bytes())
        or replay_report.get("report_hash") != hash_json(committed)
    ):
        raise ValueError("unauthenticated in-process reset replay")
    if "state_probe_hash" in replay_report:
        probe_path = folder / "reset_state_probe.npz"
        if replay_report["state_probe_hash"] != hash_bytes(probe_path.read_bytes()):
            raise ValueError("unauthenticated reset body-state probe")
        with np.load(probe_path, allow_pickle=False) as probe:
            base_keys = {
                "root_before",
                "root_after",
                "joint_before",
                "joint_after",
                "target_before",
                "target_after",
            }
            velocity_keys = {
                "root_velocity_before",
                "root_velocity_after",
                "joint_velocity_before",
                "joint_velocity_after",
            }
            if set(probe.files) not in (base_keys, base_keys | velocity_keys) or any(
                not np.isfinite(probe[key]).all() for key in probe.files
            ):
                raise ValueError("invalid reset body-state probe")
            state_checks = [
                ("root", 7, "initial_root_max_diff_m"),
                ("joint", 29, "initial_joint_max_diff_rad"),
                ("target", 29, "initial_target_max_diff_rad"),
            ]
            if velocity_keys <= set(probe.files):
                state_checks.extend(
                    [
                        ("root_velocity", 6, "initial_root_velocity_max_diff_m_s"),
                        ("joint_velocity", 29, "initial_joint_velocity_max_diff_rad_s"),
                    ]
                )
            for name, width, report_key in state_checks:
                before, after = probe[f"{name}_before"], probe[f"{name}_after"]
                expected_shape = (source["frames"], len(source["environments"]), width)
                if before.shape != expected_shape or after.shape != expected_shape:
                    raise ValueError("reset body-state shape changed")
                maximum = float(np.max(np.abs(before[0] - after[0])))
                if abs(replay_report.get(report_key, -1) - maximum) > 1e-8:
                    raise ValueError("reset initial body state disagrees with probe")
    with (
        np.load(folder / "trace.npz", allow_pickle=False) as original,
        np.load(replay_trace, allow_pickle=False) as replay,
    ):
        if set(replay.files) != set(original.files):
            raise ValueError("reset trace schema changed")
        position, force = original["ball_position_m"], original["ball_body_contact_force_peak_n"]
        replay_position, replay_force = (
            replay["ball_position_m"],
            replay["ball_body_contact_force_peak_n"],
        )
        if (
            replay_position.shape != position.shape
            or replay_force.shape != force.shape
            or any(not np.isfinite(replay[key]).all() for key in replay.files)
        ):
            raise ValueError("invalid reset physics trajectory")
    first_frames = []
    replay_first_frames = []
    body_classes_equal = True
    precontact_max_diff = 0.0
    for index in range(position.shape[1]):
        first = np.flatnonzero(np.max(force[:, index], axis=1) > 1.0)
        replay_first = np.flatnonzero(np.max(replay_force[:, index], axis=1) > 1.0)
        first_frames.append(int(first[0]) if len(first) else None)
        replay_first_frames.append(int(replay_first[0]) if len(replay_first) else None)
        body_classes_equal &= bool(
            np.array_equal(
                np.max(force[:, index], axis=0) > 1.0,
                np.max(replay_force[:, index], axis=0) > 1.0,
            )
        )
        cutoff = min(
            first[0] if len(first) else position.shape[0],
            replay_first[0] if len(replay_first) else position.shape[0],
        )
        if cutoff:
            precontact_max_diff = max(
                precontact_max_diff,
                float(np.max(np.abs(position[:cutoff, index] - replay_position[:cutoff, index]))),
            )
    max_diff = float(np.max(np.abs(position - replay_position)))
    if (
        replay_report.get("first_contact_frames") != first_frames
        or replay_report.get("replay_first_contact_frames") != replay_first_frames
        or replay_report.get("body_classes_equal") is not body_classes_equal
        or abs(replay_report.get("precontact_max_ball_position_diff_m", -1) - precontact_max_diff)
        > 1e-8
        or abs(replay_report.get("full_ball_position_max_diff_m", -1) - max_diff) > 1e-8
        or replay_report.get("minimum_replay_pelvis_z_m", 0) < 0.65
        or replay_report.get("reset_verified")
        is not bool(
            body_classes_equal
            and first_frames == replay_first_frames
            and precontact_max_diff < 0.005
        )
    ):
        raise ValueError("reset report disagrees with paired physics trace")
    return {
        "source_audit_hash": baseline["report_hash"],
        "reset_report_hash": replay_report["report_hash"],
        "reset_verified": replay_report["reset_verified"],
        "independent_new_course_count": 0,
        "precontact_max_ball_position_diff_m": precontact_max_diff,
        "full_ball_position_max_diff_m": max_diff,
    }


def audit_first_touch_candidate_execution(
    folder: Path, *, parent_folder: Path, candidate_path: Path
) -> dict[str, Any]:
    """Recompute bounded candidate outcomes from physics, never from video."""

    from rosclaw_soccer.rsi.first_touch_candidate import (
        JOINT_NAMES,
        load_first_touch_candidate,
    )

    parent_audit = audit_vector_first_touch(parent_folder)
    parent = load_physical_report(parent_folder / "report.json")
    report = load_physical_report(folder / "report.json")
    trace_path = folder / "trace.npz"
    committed = {key: value for key, value in report.items() if key != "report_hash"}
    if (
        report.get("schema")
        not in {
            "rsi_isaac_vector_first_touch_candidate_execution_v1",
            "rsi_isaac_vector_first_touch_candidate_execution_v2",
        }
        or report.get("activation_ceiling") != "SIM_ONLY"
        or report.get("learning_authorized") is not False
        or report.get("promotion_authorized") is not False
        or report.get("trained_actor") is not False
        or report.get("parent_report_hash") != parent["report_hash"]
        or report.get("frames") != parent.get("frames")
        or report.get("navigation_speed_mps", 1.4) != parent.get("navigation_speed_mps", 1.4)
        or report.get("near_ball_gap_m") != parent.get("near_ball_gap_m")
        or report.get("near_ball_speed_mps") != parent.get("near_ball_speed_mps")
        or report.get("near_ball_incoming_only", False)
        is not parent.get("near_ball_incoming_only", False)
        or report.get("asset_hash") != parent.get("asset_hash")
        or report.get("sonic_qualification_hash") != parent.get("sonic_qualification_hash")
        or report.get("onnx_graph_encoder_layout", False)
        is not parent.get("onnx_graph_encoder_layout", False)
        or report.get("torch_batch_drive", False) is not parent.get("torch_batch_drive", False)
        or report.get("torch_batch_plan_only", False)
        is not parent.get("torch_batch_plan_only", False)
        or report.get("training_course_seed") != parent.get("training_course_seed")
        or report.get("course_catalog_hash") != parent.get("course_catalog_hash")
        or (
            report.get("torch_batch_plan_only", False)
            and (
                type(report.get("torch_batch_max_internal_target_difference_rad"))
                not in (int, float)
                or not np.isfinite(report["torch_batch_max_internal_target_difference_rad"])
                or not 0 <= report["torch_batch_max_internal_target_difference_rad"] <= 1e-3
            )
        )
        or report.get("candidate_action_joint_names") != list(JOINT_NAMES)
        or report.get("report_hash") != hash_json(committed)
        or report.get("trace_hash") != hash_bytes(trace_path.read_bytes())
    ):
        raise ValueError("unauthenticated first-touch candidate physics")
    courses = tuple(
        (
            row["course"]["ball_x_m"],
            row["course"]["ball_y_local_m"],
            row["course"]["ball_vx_m_s"],
        )
        for row in parent["environments"]
    )
    candidate = load_first_touch_candidate(
        candidate_path,
        expected_courses=courses,
        parent_report_hash=parent["report_hash"],
    )
    projection_counts = report.get("candidate_action_projection_count")
    if report["schema"].endswith("_v1"):
        if projection_counts is not None:
            raise ValueError("legacy candidate report cannot claim projection audit")
        projection_counts = [0] * len(courses)
    if (
        report["candidate_hash"] != candidate.candidate_hash
        or report["candidate_actions_rad"] != [list(row) for row in candidate.actions_rad]
        or len(report["environments"]) != len(courses)
        or not isinstance(report.get("candidate_actions_applied_frames"), list)
        or len(report["candidate_actions_applied_frames"]) != len(courses)
        or any(
            type(count) is not int or not 0 <= count <= report["frames"]
            for count in report["candidate_actions_applied_frames"]
        )
        or not isinstance(projection_counts, list)
        or len(projection_counts) != len(courses)
        or any(
            type(count) is not int or not 0 <= count <= report["frames"] * len(JOINT_NAMES)
            for count in projection_counts
        )
    ):
        raise ValueError("candidate action execution ledger disagrees with manifest")
    with np.load(trace_path, allow_pickle=False) as archive:
        if set(archive.files) != {
            "ball_position_m",
            "ball_angular_velocity_rad_s",
            "ball_body_contact_force_peak_n",
        }:
            raise ValueError("candidate physics trace contract changed")
        positions = archive["ball_position_m"]
        spin = archive["ball_angular_velocity_rad_s"]
        force = archive["ball_body_contact_force_peak_n"]
    frames, count = report["frames"], len(courses)
    if (
        positions.shape != (frames, count, 3)
        or spin.shape != (frames, count, 3)
        or force.shape != (frames, count, 6)
        or not np.isfinite(positions).all()
        or not np.isfinite(spin).all()
        or not np.isfinite(force).all()
        or np.any(force < 0)
    ):
        raise ValueError("candidate physics trace invalid")
    changed_command_frames = audit_body_trace(folder, report, frames, count)
    clean, parent_clean = 0, parent_audit["clean_foot_only_episode_count"]
    rewards = []
    for index, row in enumerate(report["environments"]):
        expected = parent["environments"][index]
        if row["course"] != expected["course"] or row["lane_y_m"] != expected["lane_y_m"]:
            raise ValueError("candidate course or lane differs from Parent")
        active = np.flatnonzero(np.max(force[:, index], axis=1) > 1.0)
        first = int(active[0]) if len(active) else None
        bodies = np.flatnonzero(np.max(force[:, index], axis=0) > 1.0).tolist()
        if (
            row["first_contact_frame"] != first
            or row["contact_body_indices"] != bodies
            or not np.allclose(
                row["ball_final_local_xyz_m"],
                (
                    positions[-1, index, 0],
                    positions[-1, index, 1] - row["lane_y_m"],
                    positions[-1, index, 2],
                ),
                atol=1e-5,
                rtol=0,
            )
        ):
            raise ValueError("candidate contact evidence disagrees with physics")
        safe = row["minimum_pelvis_z_m"] >= 0.65
        foot_only = bool(bodies and set(bodies) <= {0, 1} and safe)
        clean += foot_only
        base_reward = 1.0 if foot_only else -2.0 if not safe else -1.0 if bodies else -0.5
        rewards.append(base_reward - min(0.5, projection_counts[index] * 0.01))
    result = {
        "schema": "rsi_isaac_first_touch_candidate_audit_v1",
        "activation_ceiling": "SIM_ONLY",
        "partition": "CONSUMED_DEV",
        "parent_report_hash": parent["report_hash"],
        "candidate_hash": candidate.candidate_hash,
        "execution_report_hash": report["report_hash"],
        "independent_training_course_count": count,
        "parent_clean_foot_only_count": parent_clean,
        "candidate_clean_foot_only_count": clean,
        "reward_per_course": rewards,
        "candidate_action_projection_count": projection_counts,
        "fresh_opened": False,
        "promotion_authorized": False,
    }
    if changed_command_frames is not None:
        result["recorded_body_frames"] = frames
        result["changed_navigation_command_frames"] = changed_command_frames
    result["report_hash"] = hash_json(result)
    return result


def audit_temporal_first_touch_execution(
    folder: Path, *, parent_folder: Path, candidate_path: Path
) -> dict[str, Any]:
    """Recompute each proprioceptive action from pre-step body/ball evidence."""
    from rosclaw_soccer.rsi.temporal_first_touch_policy import (
        JOINT_NAMES as TEMPORAL_JOINT_NAMES,
    )
    from rosclaw_soccer.rsi.temporal_first_touch_policy import (
        MAX_FOLLOWTHROUGH_FRAMES,
        followthrough_residual,
        load_candidate,
        temporal_residual,
    )

    parent_audit = audit_vector_first_touch(parent_folder)
    parent = load_physical_report(parent_folder / "report.json")
    report = load_physical_report(folder / "report.json")
    trace_path = folder / "trace.npz"
    courses = tuple(
        (
            row["course"]["ball_x_m"],
            row["course"]["ball_y_local_m"],
            row["course"]["ball_vx_m_s"],
        )
        for row in parent["environments"]
    )
    candidate = load_candidate(
        candidate_path, expected_courses=courses, parent_report_hash=parent["report_hash"]
    )
    followthrough_frames = report.get("temporal_followthrough_frames", 0)
    if (
        report.get("schema") != "rsi_isaac_vector_first_touch_temporal_candidate_v1"
        or report.get("activation_ceiling") != "SIM_ONLY"
        or report.get("learning_authorized") is not False
        or report.get("promotion_authorized") is not False
        or report.get("trained_actor") is not False
        or report.get("parent_report_hash") != parent["report_hash"]
        or report.get("candidate_hash") != candidate.candidate_hash
        or report.get("asset_hash") != parent["asset_hash"]
        or report.get("sonic_qualification_hash") != parent["sonic_qualification_hash"]
        or report.get("frames") != parent["frames"]
        or report.get("training_course_seed") != parent["training_course_seed"]
        or report.get("course_catalog_hash") != parent["course_catalog_hash"]
        or report.get("torch_batch_plan_only") is not True
        or report.get("navigation_speed_mps", 1.4) != 1.4
        or report.get("temporal_policy_joint_names") != list(TEMPORAL_JOINT_NAMES)
        or type(followthrough_frames) is not int
        or not 0 <= followthrough_frames <= MAX_FOLLOWTHROUGH_FRAMES
        or "near_ball_gap_m" in report
        or not isinstance(report.get("body_trace_hash"), str)
        or report.get("trace_hash") != hash_bytes(trace_path.read_bytes())
        or report.get("report_hash")
        != hash_json({key: value for key, value in report.items() if key != "report_hash"})
    ):
        raise ValueError("unauthenticated temporal first-touch physics")
    frames, count = report["frames"], len(courses)
    changed_commands = audit_body_trace(folder, report, frames, count)
    if changed_commands != 0:
        raise ValueError("temporal actor changed frozen navigation command")
    with (
        np.load(trace_path, allow_pickle=False) as physics,
        np.load(folder / "body_trace.npz", allow_pickle=False) as body,
    ):
        positions = physics["ball_position_m"]
        force = physics["ball_body_contact_force_peak_n"]
        ball_before = body["ball_position_before_step_m"]
        ball_velocity = body["ball_linear_velocity_before_step_m_s"]
        root = body["root_pose_xyzw_m"]
        joint = body["joint_position_rad"]
        joint_velocity = body["joint_velocity_rad_s"]
        applied = body["applied_residual_rad"]
        if (
            positions.shape != (frames, count, 3)
            or force.shape != (frames, count, 6)
            or not np.allclose(ball_before[1:], positions[:-1], atol=1e-5, rtol=0)
        ):
            raise ValueError("temporal pre-step ball/body alignment invalid")
        clean = 0
        rewards = []
        applied_frames = []
        for i, (x, y, _vx) in enumerate(courses):
            row = report["environments"][i]
            active = np.flatnonzero(np.max(force[:, i], axis=1) > 1.0)
            first = int(active[0]) if len(active) else None
            bodies = np.flatnonzero(np.max(force[:, i], axis=0) > 1.0).tolist()
            if (
                row["course"] != parent["environments"][i]["course"]
                or row["lane_y_m"] != parent["environments"][i]["lane_y_m"]
                or row["first_contact_frame"] != first
                or row["contact_body_indices"] != bodies
                or not np.allclose(ball_before[0, i], (x, row["lane_y_m"] + y, 0.13), atol=1e-6)
                or not np.allclose(
                    row["ball_final_local_xyz_m"],
                    (
                        positions[-1, i, 0],
                        positions[-1, i, 1] - row["lane_y_m"],
                        positions[-1, i, 2],
                    ),
                    atol=1e-5,
                    rtol=0,
                )
                or row["minimum_pelvis_z_m"] < 0.65
            ):
                raise ValueError("temporal course, body or contact ledger invalid")
            count_applied = 0
            for frame in range(frames):
                if first is not None and frame > first:
                    expected = (
                        followthrough_residual(
                            applied[first, i],
                            elapsed_frames=frame - first,
                            followthrough_frames=followthrough_frames,
                        )
                        if followthrough_frames
                        else np.zeros(len(TEMPORAL_JOINT_NAMES))
                    )
                else:
                    expected = temporal_residual(
                        candidate.weights_per_course[i],
                        ball_relative_xyz_m=(
                            float(ball_before[frame, i, 0] - root[frame, i, 0]),
                            float(ball_before[frame, i, 1] - root[frame, i, 1]),
                            float(ball_before[frame, i, 2] - root[frame, i, 2]),
                        ),
                        ball_vx_m_s=float(ball_velocity[frame, i, 0]),
                        joint_position_rad=joint[frame, i],
                        joint_velocity_rad_s=joint_velocity[frame, i],
                    )
                actual = applied[frame, i]
                if np.any(expected):
                    count_applied += 1
                if np.any((np.abs(actual - expected) > 1e-5) & (np.abs(actual) > 1e-5)):
                    raise ValueError("temporal action differs from committed body-aware policy")
            applied_frames.append(count_applied)
            foot_only = bool(bodies and set(bodies) <= {0, 1})
            clean += foot_only
            rewards.append(1.0 if foot_only else -1.0 if bodies else -0.5)
    projections = report.get("temporal_policy_projection_count")
    if (
        report.get("temporal_policy_applied_frames") != applied_frames
        or not isinstance(projections, list)
        or len(projections) != count
        or any(type(value) is not int or not 0 <= value <= frames * 3 for value in projections)
    ):
        raise ValueError("temporal action application or projection ledger invalid")
    result: dict[str, Any] = {
        "schema": "rsi_isaac_temporal_first_touch_candidate_audit_v1",
        "activation_ceiling": "SIM_ONLY",
        "partition": "CONSUMED_DEV",
        "parent_report_hash": parent["report_hash"],
        "candidate_hash": candidate.candidate_hash,
        "execution_report_hash": report["report_hash"],
        "independent_training_course_count": count,
        "parent_clean_foot_only_count": parent_audit["clean_foot_only_episode_count"],
        "candidate_clean_foot_only_count": clean,
        "reward_per_course": rewards,
        "temporal_policy_applied_frames": applied_frames,
        "temporal_policy_projection_count": projections,
        "recorded_body_frames": frames,
        "fresh_opened": False,
        "promotion_authorized": False,
    }
    result["report_hash"] = hash_json(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--reset-replay", action="store_true")
    parser.add_argument("--candidate-manifest", type=Path)
    parser.add_argument("--temporal-candidate-manifest", type=Path)
    parser.add_argument("--parent-folder", type=Path)
    args = parser.parse_args()
    if args.reset_replay and (
        args.candidate_manifest is not None or args.temporal_candidate_manifest is not None
    ):
        parser.error("reset replay and candidate audit are separate protocols")
    if args.candidate_manifest is not None and args.temporal_candidate_manifest is not None:
        parser.error("choose one candidate protocol")
    if (args.candidate_manifest is not None or args.temporal_candidate_manifest is not None) != (
        args.parent_folder is not None
    ):
        parser.error("candidate audit requires both manifest and Parent folder")
    report = (
        audit_temporal_first_touch_execution(
            args.folder,
            parent_folder=args.parent_folder,
            candidate_path=args.temporal_candidate_manifest,
        )
        if args.temporal_candidate_manifest is not None and args.parent_folder is not None
        else audit_first_touch_candidate_execution(
            args.folder,
            parent_folder=args.parent_folder,
            candidate_path=args.candidate_manifest,
        )
        if args.candidate_manifest is not None and args.parent_folder is not None
        else audit_reset_replay(args.folder)
        if args.reset_replay
        else audit_vector_first_touch(args.folder)
    )
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
