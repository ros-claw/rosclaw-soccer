"""Recompute a shared-world shadow gate from committed raw observation and physics."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.providers.g1.asset_qualification import trajectory_digest
from rosclaw_soccer.rsi.contact_time_phase_features import (
    current_context,
    gait_phase_features,
    predict_contact_time,
)
from rosclaw_soccer.rsi.late_swing_memory import load_late_swing_actor
from rosclaw_soccer.rsi.taskspace_gate_memory import select_taskspace_gate
from rosclaw_soccer.rsi.team_memory_support import (
    leave_one_out_support_radius,
    query_support_distance,
)
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.skills.team.foot_kinematics import TeamFootKinematics


def audit_shadow(folder: Path, protocol_path: Path, actor_path: Path) -> dict[str, Any]:
    report = json.loads((folder / "report.json").read_text(encoding="utf-8"))
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    actor = load_late_swing_actor(actor_path)
    sample = report["sample"]
    trace_path = folder / "trajectory.npz"
    if (
        report.get("schema") != "rsi_team_late_swing_shadow_report_v12b"
        or protocol.get("schema") != "rsi_team_late_swing_shadow_protocol_v12b"
        or report.get("report_hash")
        != hash_json({k: v for k, v in report.items() if k != "report_hash"})
        or report.get("protocol_hash") != hash_bytes(protocol_path.read_bytes())
        or report.get("actor_hash") != actor["actor_hash"]
        or report.get("trace_hash") != hash_bytes(trace_path.read_bytes())
        or report.get("shadow_motor_target_count") != 0
        or report.get("source_stable_during_run") is not True
        or report.get("cross_asset_calibration_equivalent") is not False
        or report.get("policy_intervention_authorized") is not False
        or report.get("promotion_authorized") is not False
        or report.get("world_result", {}).get("safe") is not True
        or report.get("world_result", {}).get("motor_fault_agents")
    ):
        raise ValueError("unbound or unsafe shadow report")
    source_root = Path(__file__).parents[1]
    for relative, digest in report["source_hashes"].items():
        path = (source_root / relative).resolve()
        if (
            not path.is_relative_to(source_root.resolve())
            or hash_bytes(path.read_bytes()) != digest
        ):
            raise ValueError("shadow source changed")
    q = np.asarray(sample["raw_qpos"], dtype=float)
    v = np.asarray(sample["raw_qvel"], dtype=float)
    if q.shape != (43,) or v.shape != (41,) or not np.isfinite(q).all() or not np.isfinite(v).all():
        raise ValueError("raw same-frame shadow state missing")
    TeamFootKinematics(
        agent_id=sample["agent_id"],
        frame=sample["frame"],
        foot_position_world_m=tuple(tuple(row) for row in sample["foot_position_world_m"]),
        foot_linear_jacobian_world=tuple(
            tuple(tuple(row) for row in side) for side in sample["foot_linear_jacobian_world"]
        ),
        leg_joint_limits_rad=tuple(
            tuple(tuple(row) for row in side) for side in sample["leg_joint_limits_rad"]
        ),
    )
    frame = sample["frame"]
    key = sample["agent_id"].replace(".", "_")
    with np.load(trace_path, allow_pickle=False) as archive:
        trace = {name: archive[name] for name in archive.files}
    if (
        report["trajectory_digest"] != trajectory_digest(trace)
        or frame != protocol["sample_frame"]
        or sample["agent_id"] != protocol["shadow_agent_id"]
        or not np.array_equal(q[:7], trace[f"{key}_pelvis_pose"][frame - 1])
        or not np.array_equal(q[36:43], trace["ball_pose"][frame - 1])
        or not np.array_equal(v[35:38], trace["ball_velocity"][frame - 1, :3])
        or not np.array_equal(
            np.asarray(sample["foot_position_world_m"])[0],
            trace[f"{key}_left_foot_position"][frame - 1],
        )
        or not np.array_equal(
            np.asarray(sample["foot_position_world_m"])[1],
            trace[f"{key}_right_foot_position"][frame - 1],
        )
    ):
        raise ValueError("shadow raw state differs from committed physical trajectory")
    raw = current_context(q[:7][None], v[:6][None], q[36:39][None], v[35:38][None])
    predicted = predict_contact_time(raw, np.asarray(actor["contact_time_weights"]))
    features = gait_phase_features(raw, predicted)
    memories = np.asarray(actor["memory_features"])
    nearest = query_support_distance(memories, features[0])
    radius = leave_one_out_support_radius(memories)
    gate = select_taskspace_gate(
        features,
        memories,
        np.asarray(actor["memory_clean"]),
        np.asarray(actor["memory_reward"]),
        np.asarray(actor["memory_groups"]),
        neighbors=actor["neighbors"],
        confidence=actor["confidence"],
        baseline_clean_ceiling=actor["baseline_clean_ceiling"],
    )
    qw, qx, qy, qz = q[3:7]
    yaw = math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
    direction = bool(abs(yaw) <= 0.35 and v[35] < -0.05)
    supported = bool(nearest <= radius + 1e-12)
    if (
        not np.allclose(sample["features"], features[0], atol=1e-12, rtol=0)
        or not math.isclose(
            sample["predicted_contact_offset_frames"], float(predicted[0]), abs_tol=1e-12
        )
        or not math.isclose(sample["nearest_training_memory_distance"], nearest, abs_tol=1e-12)
        or not math.isclose(sample["training_support_radius"], radius, abs_tol=1e-12)
        or sample["within_frozen_training_support"] is not supported
        or sample["raw_memory_gate_selected"] is not bool(gate[0])
        or sample["course_direction_supported"] is not direction
        or sample["effective_shadow_gate_selected"]
        is not (bool(gate[0]) and supported and direction)
        or sample["policy_intervention_authorized"] is not False
    ):
        raise ValueError("shadow gate differs from raw same-frame state")
    return {
        "schema": "rsi_team_late_swing_shadow_audit_v12b",
        "source_report_hash": report["report_hash"],
        "actor_hash": actor["actor_hash"],
        "training_support_radius": radius,
        "query_distance": nearest,
        "raw_gate_selected": bool(gate[0]),
        "within_training_support": supported,
        "effective_gate_selected": bool(gate[0]) and supported and direction,
        "policy_intervention_authorized": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", required=True, type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--actor", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(audit_shadow(args.folder, args.protocol, args.actor), sort_keys=True))


if __name__ == "__main__":
    main()
