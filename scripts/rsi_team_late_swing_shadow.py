"""Read-only late-swing actor domain probe in one shared MuJoCo team world."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.providers.g1.asset_qualification import (
    qualify_g1_assets,
    trajectory_digest,
)
from rosclaw_soccer.rsi import contact_time_phase_features as time_phase_module
from rosclaw_soccer.rsi import late_swing_memory as late_swing_module
from rosclaw_soccer.rsi import taskspace_gate_memory as taskspace_gate_module
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
from rosclaw_soccer.skills.team.independent_team_world import (
    IndependentTeamWorldConfig,
    IndependentTeamWorldScenario,
    simulate_independent_team_world,
)
from rosclaw_soccer.skills.team.motor_option import TeamMotorObservation
from rosclaw_soccer.training.independent_team_growth import build_independent_three_vs_three_fixture


class ReadOnlyLateSwingShadow:
    needs_foot_kinematics = True

    def __init__(self, actor: dict[str, Any], agent_id: str, frame: int) -> None:
        self.actor = actor
        self.agent_id = agent_id
        self.frame = frame
        self.next_frame = 0
        self.motor_target_count = 0
        self.measurement: dict[str, Any] | None = None
        self.contract_hash = hash_json(
            {
                "schema": "rsi_team_late_swing_readonly_shadow_v1",
                "actor": actor["actor_hash"],
                "agent": agent_id,
                "frame": frame,
                "target_count": 0,
            }
        )

    def propose(self, observation: TeamMotorObservation) -> None:
        if (
            observation.agent_id != self.agent_id
            or observation.frame != self.next_frame
            or observation.foot_kinematics is None
        ):
            raise ValueError("shadow requires consecutive same-player measured kinematics")
        self.next_frame += 1
        if observation.frame != self.frame:
            return None
        q = np.asarray(observation.qpos, dtype=float)
        v = np.asarray(observation.qvel, dtype=float)
        root = q[:7][None, :]
        ball = q[36:39][None, :]
        root_velocity = v[:6][None, :]
        ball_velocity = v[35:38][None, :]
        raw = current_context(root, root_velocity, ball, ball_velocity)
        predicted = predict_contact_time(raw, np.asarray(self.actor["contact_time_weights"]))
        features = gait_phase_features(raw, predicted)
        gate = select_taskspace_gate(
            features,
            np.asarray(self.actor["memory_features"]),
            np.asarray(self.actor["memory_clean"]),
            np.asarray(self.actor["memory_reward"]),
            np.asarray(self.actor["memory_groups"]),
            neighbors=self.actor["neighbors"],
            confidence=self.actor["confidence"],
            baseline_clean_ceiling=self.actor["baseline_clean_ceiling"],
        )
        qw, qx, qy, qz = q[3:7]
        yaw = math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
        memories = np.asarray(self.actor["memory_features"])
        nearest = query_support_distance(memories, features[0])
        support_radius = leave_one_out_support_radius(memories)
        supported = bool(nearest <= support_radius + 1e-12)
        feet = np.asarray(observation.foot_kinematics.foot_position_world_m)
        self.measurement = {
            "frame": observation.frame,
            "agent_id": observation.agent_id,
            "root_pose_m": root[0, :3].tolist(),
            "ball_position_m": ball[0].tolist(),
            "ball_velocity_m_s": ball_velocity[0].tolist(),
            "foot_position_world_m": feet.tolist(),
            "foot_linear_jacobian_world": observation.foot_kinematics.foot_linear_jacobian_world,
            "leg_joint_limits_rad": observation.foot_kinematics.leg_joint_limits_rad,
            "raw_qpos": q.tolist(),
            "raw_qvel": v.tolist(),
            "foot_ball_distance_m": np.linalg.norm(feet - ball, axis=1).tolist(),
            "features": features[0].tolist(),
            "predicted_contact_offset_frames": float(predicted[0]),
            "nearest_training_memory_distance": nearest,
            "training_support_radius": support_radius,
            "within_frozen_training_support": supported,
            "raw_memory_gate_selected": bool(gate[0]),
            "heading_yaw_rad": yaw,
            "course_direction_supported": bool(abs(yaw) <= 0.35 and ball_velocity[0, 0] < -0.05),
            "effective_shadow_gate_selected": bool(
                gate[0] and supported and abs(yaw) <= 0.35 and ball_velocity[0, 0] < -0.05
            ),
            "policy_intervention_authorized": False,
        }
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", required=True, type=Path)
    parser.add_argument("--actor", required=True, type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("output directory already exists")
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    actor = load_late_swing_actor(args.actor)
    if (
        actor["policy_source_hash"] != hash_bytes(Path(taskspace_gate_module.__file__).read_bytes())
        or actor["feature_source_hash"] != hash_bytes(Path(time_phase_module.__file__).read_bytes())
        or actor["loader_source_hash"] != hash_bytes(Path(late_swing_module.__file__).read_bytes())
    ):
        raise ValueError("late-swing actor implementation changed")
    if (
        protocol.get("schema") != "rsi_team_late_swing_shadow_protocol_v12b"
        or protocol.get("actor_hash") != actor["actor_hash"]
        or protocol.get("policy_intervention_authorized") is not False
        or protocol.get("shadow_motor_target_count_required") != 0
    ):
        raise ValueError("uncommitted read-only shadow protocol")
    qualification = qualify_g1_assets(args.asset_root)
    qualification.require_eligible()
    fixture = build_independent_three_vs_three_fixture(args.asset_root)
    agent_id = protocol["shadow_agent_id"]
    if agent_id not in {player.agent_id for player in fixture.players}:
        raise ValueError("shadow player is not in the qualified roster")
    shadow = ReadOnlyLateSwingShadow(actor, agent_id, protocol["sample_frame"])
    source_paths = (
        Path(__file__),
        Path(__file__).parents[1] / "src/rosclaw_soccer/skills/team/independent_team_world.py",
        Path(__file__).parents[1] / "src/rosclaw_soccer/skills/team/motor_option.py",
        Path(__file__).parents[1] / "src/rosclaw_soccer/skills/team/foot_kinematics.py",
        Path(__file__).parents[1] / "src/rosclaw_soccer/rsi/team_memory_support.py",
    )
    source_hashes = {
        str(path.relative_to(Path(__file__).parents[1])): hash_bytes(path.read_bytes())
        for path in source_paths
    }
    course = protocol["scenario"]
    scenario = IndependentTeamWorldScenario(
        course["scenario_id"],
        tuple(course["ball_initial_position_m"]),
        tuple(course["ball_initial_velocity_mps"]),
        course["seed"],
    )
    result, trace = simulate_independent_team_world(
        asset_root=args.asset_root,
        roster=fixture.roster,
        cells=fixture.cells,
        players=fixture.players,
        scenario=scenario,
        goal=fixture.goal,
        config=IndependentTeamWorldConfig(
            simulation_duration_sec=protocol["simulation_duration_sec"]
        ),
        motor_options={agent_id: shadow},
    )
    if shadow.measurement is None or shadow.motor_target_count != 0:
        raise ValueError("shadow measurement missing or shadow attempted motor output")
    if source_hashes != {
        str(path.relative_to(Path(__file__).parents[1])): hash_bytes(path.read_bytes())
        for path in source_paths
    }:
        raise ValueError("shadow implementation changed during physics execution")
    args.output_dir.mkdir(parents=True)
    trace_path = args.output_dir / "trajectory.npz"
    np.savez_compressed(trace_path, **trace)  # type: ignore[arg-type]
    report = {
        "schema": "rsi_team_late_swing_shadow_report_v12b",
        "activation_ceiling": "SIM_ONLY",
        "protocol_hash": hash_bytes(args.protocol.read_bytes()),
        "runner_hash": hash_bytes(Path(__file__).read_bytes()),
        "source_hashes": source_hashes,
        "source_stable_during_run": True,
        "actor_hash": actor["actor_hash"],
        "asset_body_hash": qualification.body_hash,
        "fixture_hash": fixture.fixture_hash,
        "scenario_hash": scenario.scenario_hash,
        "shadow_motor_contract_hash": shadow.contract_hash,
        "shadow_motor_target_count": shadow.motor_target_count,
        "trace_hash": hash_bytes(trace_path.read_bytes()),
        "trajectory_digest": trajectory_digest(trace),
        "sample": shadow.measurement,
        "world_result": result.to_dict(),
        "cross_asset_calibration_equivalent": False,
        "policy_intervention_authorized": False,
        "promotion_authorized": False,
    }
    report["report_hash"] = hash_json(report)
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("RSI_TEAM_LATE_SWING_SHADOW=" + json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
