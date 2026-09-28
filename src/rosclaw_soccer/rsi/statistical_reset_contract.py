"""Audit reset safety without requiring bitwise PhysX replay identity.

This permits *designing* stochastic RL episodes, not promoting or executing
learned robot motion. Each reset must restore the observable initial state,
clear contact history, and preserve body safety despite solver variation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.rsi.vector_first_touch_evidence import audit_reset_replay
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json

STATE_WIDTHS = {
    "root": 7,
    "root_velocity": 6,
    "joint": 29,
    "joint_velocity": 29,
    "target": 29,
}
TRACE_KEYS = {
    "ball_position_m",
    "ball_angular_velocity_rad_s",
    "ball_body_contact_force_peak_n",
}


def audit_statistical_reset(folder: Path) -> dict[str, Any]:
    """Require exact reset starts and contact clearing; allow later trajectory drift."""

    first_audit = audit_reset_replay(folder)
    base = json.loads((folder / "report.json").read_text(encoding="utf-8"))
    first = json.loads((folder / "reset_report.json").read_text(encoding="utf-8"))
    second_path = folder / "second_reset_report.json"
    second = json.loads(second_path.read_text(encoding="utf-8"))
    second_body = {key: value for key, value in second.items() if key != "report_hash"}
    if (
        second.get("schema") != "rsi_isaac_vector_first_touch_second_reset_v1"
        or second.get("activation_ceiling") != "SIM_ONLY"
        or second.get("learning_authorized") is not False
        or second.get("promotion_authorized") is not False
        or second.get("first_reset_report_hash") != first["report_hash"]
        or second.get("report_hash") != hash_json(second_body)
        or second.get("second_trace_hash")
        != hash_bytes((folder / "second_reset_replay.npz").read_bytes())
        or second.get("second_state_hash")
        != hash_bytes((folder / "second_reset_state_probe.npz").read_bytes())
    ):
        raise ValueError("unauthenticated second physical reset")
    frames = base["frames"]
    env_count = len(base["environments"])
    with (
        np.load(folder / "reset_state_probe.npz", allow_pickle=False) as state,
        np.load(folder / "second_reset_state_probe.npz", allow_pickle=False) as second_state,
        np.load(folder / "trace.npz", allow_pickle=False) as base_trace,
        np.load(folder / "reset_replay.npz", allow_pickle=False) as first_trace,
        np.load(folder / "second_reset_replay.npz", allow_pickle=False) as second_trace,
    ):
        if (
            set(state.files)
            != {f"{name}_{suffix}" for name in STATE_WIDTHS for suffix in ("before", "after")}
            or set(second_state.files) != set(STATE_WIDTHS)
            or any(
                set(trace.files) != TRACE_KEYS for trace in (base_trace, first_trace, second_trace)
            )
        ):
            raise ValueError("reset state or trace schema changed")
        initial_differences = []
        for name, width in STATE_WIDTHS.items():
            expected = (frames, env_count, width)
            before = state[f"{name}_before"]
            after = state[f"{name}_after"]
            again = second_state[name]
            if (
                before.shape != expected
                or after.shape != expected
                or again.shape != expected
                or not np.isfinite(before).all()
                or not np.isfinite(after).all()
                or not np.isfinite(again).all()
            ):
                raise ValueError("reset body state invalid")
            initial_differences.extend(
                [
                    float(np.max(np.abs(before[0] - after[0]))),
                    float(np.max(np.abs(before[0] - again[0]))),
                ]
            )
        ball_initial_differences = []
        contact_frames = []
        contact_bodies = []
        for trace in (base_trace, first_trace, second_trace):
            positions = trace["ball_position_m"]
            spin = trace["ball_angular_velocity_rad_s"]
            forces = trace["ball_body_contact_force_peak_n"]
            if (
                positions.shape != (frames, env_count, 3)
                or spin.shape != positions.shape
                or forces.shape != (frames, env_count, 6)
                or any(not np.isfinite(trace[key]).all() for key in TRACE_KEYS)
                or np.any(forces < 0)
            ):
                raise ValueError("reset ball physics invalid")
            if trace is not base_trace:
                ball_initial_differences.append(
                    float(np.max(np.abs(base_trace["ball_position_m"][0] - positions[0])))
                )
                ball_initial_differences.append(
                    float(np.max(np.abs(base_trace["ball_angular_velocity_rad_s"][0] - spin[0])))
                )
            active = np.max(forces, axis=2) > 1.0
            first_frames = [
                int(indices[0]) if len(indices) else None
                for indices in (np.flatnonzero(active[:, i]) for i in range(env_count))
            ]
            if np.any(forces[:20] > 1.0):
                raise ValueError("stale or immediate post-reset contact")
            contact_frames.append(first_frames)
            contact_bodies.append(
                [
                    np.flatnonzero(np.max(forces[:, index], axis=0) > 1.0).tolist()
                    for index in range(env_count)
                ]
            )
        first_pelvis_min = float(np.min(state["root_after"][:, :, 2]))
        second_pelvis_min = float(np.min(second_state["root"][:, :, 2]))
    max_initial_difference = max(initial_differences + ball_initial_differences)
    if (
        max_initial_difference > 1e-6
        or second.get("first_reset_contact_frames") != contact_frames[1]
        or second.get("second_reset_contact_frames") != contact_frames[2]
        or second.get("first_reset_contact_bodies") != contact_bodies[1]
        or second.get("second_reset_contact_bodies") != contact_bodies[2]
        or first_pelvis_min < 0.65
        or second_pelvis_min < 0.65
        or abs(second.get("second_minimum_pelvis_z_m", -1) - second_pelvis_min) > 1e-6
        or abs(first.get("minimum_replay_pelvis_z_m", -1) - first_pelvis_min) > 1e-6
    ):
        raise ValueError("statistical reset initial state or body safety failed")
    report = {
        "schema": "rsi_isaac_statistical_reset_contract_v1",
        "activation_ceiling": "SIM_ONLY",
        "training_environment_design_ready": True,
        "learning_authorized": False,
        "promotion_authorized": False,
        "base_report_hash": base["report_hash"],
        "first_reset_audit_hash": first_audit["reset_report_hash"],
        "second_reset_report_hash": second["report_hash"],
        "episode_count": 3 * env_count,
        "unique_course_count": env_count,
        "max_initial_state_difference": max_initial_difference,
        "first_contact_frames": contact_frames,
        "strict_trajectory_replay_passed": first_audit["reset_verified"],
    }
    report["report_hash"] = hash_json(report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = audit_statistical_reset(args.folder)
    if args.output is not None:
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, sort_keys=True)
            stream.write("\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
