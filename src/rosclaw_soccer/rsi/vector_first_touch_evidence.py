"""Fail-closed audit of independent SIM_ONLY Isaac first-touch environments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.sim.contracts import hash_bytes, hash_json


def audit_vector_first_touch(folder: Path) -> dict[str, Any]:
    report: dict[str, Any] = json.loads((folder / "report.json").read_text(encoding="utf-8"))
    trace_path = folder / "trace.npz"
    committed = {key: value for key, value in report.items() if key != "report_hash"}
    if (
        report.get("schema") != "rsi_isaac_vector_first_touch_smoke_v1"
        or report.get("activation_ceiling") != "SIM_ONLY"
        or report.get("learning_authorized") is not False
        or report.get("promotion_authorized") is not False
        or report.get("report_hash") != hash_json(committed)
        or report.get("trace_hash") != hash_bytes(trace_path.read_bytes())
    ):
        raise ValueError("unauthenticated vector first-touch evidence")
    entries = report.get("environments")
    frames = report.get("frames")
    if (
        not isinstance(entries, list)
        or not 2 <= len(entries) <= 16
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
    lanes = np.asarray([row["lane_y_m"] for row in entries], dtype=np.float64)
    if not np.isfinite(lanes).all() or np.min(np.diff(lanes)) < 6.0:
        raise ValueError("training lanes are not physically isolated")
    if np.max(np.abs(positions[:, :, 1] - lanes[None, :])) >= 4.0:
        raise ValueError("ball escaped its lane")
    courses = []
    clean = 0
    positives = 0
    sides: set[str] = set()
    for i, row in enumerate(entries):
        course = row["course"]
        key = (course["ball_x_m"], course["ball_y_local_m"], course["ball_vx_m_s"])
        courses.append(key)
        if (
            key[0] not in (2.4, 2.6)
            or key[1] not in (-0.1, 0.1)
            or key[2] not in (-0.5, 0.5)
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
    result["report_hash"] = hash_json(result)
    return result


def audit_reset_replay(folder: Path) -> dict[str, Any]:
    """Verify paired in-process reset without counting replay as new courses."""

    baseline = audit_vector_first_touch(folder)
    source = json.loads((folder / "report.json").read_text(encoding="utf-8"))
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--reset-replay", action="store_true")
    args = parser.parse_args()
    report = (
        audit_reset_replay(args.folder)
        if args.reset_replay
        else audit_vector_first_touch(args.folder)
    )
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
