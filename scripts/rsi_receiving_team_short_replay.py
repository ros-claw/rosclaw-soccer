"""SIM_ONLY zero-intervention full-eight-G1 short-horizon replay fidelity exam."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from rosclaw_soccer.sim.contracts import (
    G1_DDS_JOINT_NAMES,
    G1_HARD_TORQUE_LIMITS,
    hash_bytes,
    hash_json,
)
from rosclaw_soccer.sim.physical_checkpoint import PhysicalCheckpoint, compiled_model_hash
from rosclaw_soccer.skills.team.independent_team_world import _project_joint_safe_torque
from rosclaw_soccer.training.continuous_match_residual_ppo import collection_fixture
from rosclaw_soccer.training.receiving_classroom import r0_receiving_configuration
from rosclaw_soccer.world.multi_player import build_g1_multi_player_stadium_model

START = 45
STOP = 100
FOCAL = "red.finisher"


def replay(*, asset_root: Path, captured: Path, output_dir: Path) -> dict[str, Any]:
    source = Path(__file__)
    root = source.parents[1]
    paths = {
        "source": source,
        "world": root / "src/rosclaw_soccer/skills/team/independent_team_world.py",
        "checkpoint": root / "src/rosclaw_soccer/sim/physical_checkpoint.py",
        "stadium": root / "src/rosclaw_soccer/world/multi_player.py",
    }
    source_hashes = {name: hash_bytes(path.read_bytes()) for name, path in paths.items()}
    if output_dir.exists() or output_dir.resolve().is_relative_to(root):
        raise ValueError("new external SIM_ONLY eight-G1 replay directory required")
    capture: dict[str, Any] = json.loads((captured / "report.json").read_text(encoding="utf-8"))
    capture_hash = capture.pop("report_hash")
    trace_path = captured / "team-motor-trace.npz"
    if (
        capture_hash != hash_json(capture)
        or capture["schema"] != "rosclaw_soccer.rsi.receiving_team_motor_capture.v1"
        or capture["read_only_replay_exact"] is not True
        or capture["checkpoint_frame"] != START
        or capture["trace_hash"] != hash_bytes(trace_path.read_bytes())
        or capture["promotion_authorized"] is not False
    ):
        raise ValueError("sealed frame-45 eight-G1 motor capture required")
    with np.load(trace_path, allow_pickle=False) as payload:
        trace = {key: np.asarray(payload[key]) for key in payload.files}
    fixture = collection_fixture(asset_root)
    world, _ = r0_receiving_configuration()
    model = build_g1_multi_player_stadium_model(
        asset_root,
        players=fixture.players,
        spec=fixture.goal,
        left_goal_plane_x_m=world.left_goal_plane_x_m if world.bilateral_goals else None,
    )
    model.opt.timestep = 0.002
    model_hash = compiled_model_hash(model)
    if model_hash != capture["model_hash"]:
        raise ValueError("full-eight-G1 compiled replay physics mismatch")
    state = np.asarray(trace["initial_integration_state"], dtype="<f8")
    checkpoint = PhysicalCheckpoint(
        model_hash=model_hash,
        mujoco_version=str(capture["mujoco_version"]),
        state_bytes=state.tobytes(),
        state_hash=str(capture["integration_hash"]),
    )
    data = checkpoint.restore(model)
    agents = {}
    for player in fixture.players:
        prefix = player.body_prefix
        joint_ids = np.asarray(
            [
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, prefix + name)
                for name in G1_DDS_JOINT_NAMES
            ],
            dtype=np.int64,
        )
        actuator_ids = np.asarray(
            [
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, prefix + name)
                for name in G1_DDS_JOINT_NAMES
            ],
            dtype=np.int64,
        )
        if np.any(joint_ids < 0) or np.any(actuator_ids < 0):
            raise ValueError("complete eight-G1 joint and motor replay identities required")
        free = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, prefix + "floating_base_joint")
        agents[player.agent_id] = {
            "joint_ids": joint_ids,
            "actuators": actuator_ids,
            "qpos": model.jnt_qposadr[joint_ids],
            "qvel": model.jnt_dofadr[joint_ids],
            "root": int(model.jnt_qposadr[free]),
        }
    ball_joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "ball_free")
    ball_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "ball_geom")
    ball_qpos = int(model.jnt_qposadr[ball_joint])
    ball_qvel = int(model.jnt_dofadr[ball_joint])
    hard = np.asarray(G1_HARD_TORQUE_LIMITS, dtype=np.float64)
    guarded = 0.85 * hard
    ball_speed = []
    ball_pelvis_distance = []
    robot_min_height = {agent: float("inf") for agent in agents}
    robot_max_tilt = {agent: 0.0 for agent in agents}
    first_own_foot_frame = None
    own_nonfoot_seen = False
    maximum_executed_torque_error_nm = 0.0
    for frame in range(START, STOP + 1):
        for substep in range(10):
            for agent, index in agents.items():
                key = agent.replace(".", "_")
                q = data.qpos[index["qpos"]]
                dq = data.qvel[index["qvel"]]
                target = trace[f"{key}_captured_pd_target"][frame]
                kp = trace[f"{key}_captured_pd_kp"][frame]
                kd = trace[f"{key}_captured_pd_kd"][frame]
                extra = trace[f"{key}_captured_extra_torque_nm"][frame * 10 + substep]
                raw = kp * (target - q) - kd * dq + extra
                projected = _project_joint_safe_torque(
                    joint_position=q,
                    joint_velocity=dq,
                    commanded_torque=raw,
                    joint_ranges=np.asarray(model.jnt_range[index["joint_ids"]]),
                    limited=model.jnt_limited[index["joint_ids"]].astype(bool),
                    margin_rad=world.joint_guard_margin_rad,
                )
                torque = np.clip(projected, -guarded, guarded)
                saved = trace[f"{key}_captured_executed_torque_nm"][frame * 10 + substep]
                maximum_executed_torque_error_nm = max(
                    maximum_executed_torque_error_nm, float(np.max(np.abs(torque - saved)))
                )
                data.ctrl[index["actuators"]] = torque
            mujoco.mj_step(model, data)
            for agent, index in agents.items():
                root_index = int(index["root"])
                robot_min_height[agent] = min(
                    robot_min_height[agent], float(data.qpos[root_index + 2])
                )
                quat = data.qpos[root_index + 3 : root_index + 7]
                tilt = float(np.arccos(np.clip(1 - 2 * (quat[1] ** 2 + quat[2] ** 2), -1, 1)))
                robot_max_tilt[agent] = max(robot_max_tilt[agent], tilt)
            for contact_id in range(data.ncon):
                contact = data.contact[contact_id]
                a, b = int(contact.geom1), int(contact.geom2)
                if ball_geom not in (a, b):
                    continue
                other = b if a == ball_geom else a
                name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, other) or ""
                if not name.startswith("red_finisher_"):
                    continue
                if "foot" in name:
                    if first_own_foot_frame is None:
                        first_own_foot_frame = frame
                else:
                    own_nonfoot_seen = True
        speed = float(np.linalg.norm(data.qvel[ball_qvel : ball_qvel + 2]))
        focal_root = int(agents[FOCAL]["root"])
        distance = float(
            np.linalg.norm(
                data.qpos[ball_qpos : ball_qpos + 2] - data.qpos[focal_root : focal_root + 2]
            )
        )
        ball_speed.append(speed)
        ball_pelvis_distance.append(distance)
    reference_speed = float(np.linalg.norm(trace["ball_velocity"][86, :2]))
    reference_distance = float(
        np.linalg.norm(trace["ball_pose"][86, :2] - trace["red_finisher_pelvis_pose"][86, :2])
    )
    foot_code = capture["agent_ids"].index(FOCAL) + 1
    reference_foot = np.flatnonzero(
        (trace["ball_contact_agent_code"] == foot_code) & (trace["ball_contact_foot_code"] > 0)
    )
    reference_first_foot = None if not len(reference_foot) else int(reference_foot[0])
    speed_error = abs(ball_speed[86 - START] - reference_speed)
    distance_error = abs(ball_pelvis_distance[86 - START] - reference_distance)
    all_safe = all(
        robot_min_height[agent] >= 0.55 and robot_max_tilt[agent] < 0.8 for agent in agents
    )
    passed = bool(
        reference_first_foot is not None
        and first_own_foot_frame is not None
        and abs(first_own_foot_frame - reference_first_foot) <= 1
        and speed_error <= 0.05
        and distance_error <= 0.05
        and not own_nonfoot_seen
        and all_safe
    )
    if {name: hash_bytes(path.read_bytes()) for name, path in paths.items()} != source_hashes:
        raise RuntimeError("short eight-G1 replay source changed during physics")
    output_dir.mkdir(parents=True)
    trajectory_path = output_dir / "short-replay.npz"
    np.savez_compressed(
        trajectory_path,
        frame=np.arange(START, STOP + 1),
        ball_speed_mps=np.asarray(ball_speed),
        ball_pelvis_distance_m=np.asarray(ball_pelvis_distance),
    )
    report: dict[str, Any] = {
        "schema": "rosclaw_soccer.rsi.receiving_team_short_replay.v1",
        "activation_ceiling": "SIM_ONLY",
        "source_hashes": source_hashes,
        "capture_report_hash": capture_hash,
        "compiled_model_hash": model_hash,
        "integration_hash": capture["integration_hash"],
        "trajectory_hash": hash_bytes(trajectory_path.read_bytes()),
        "frame_range": [START, STOP],
        "reference_first_foot_frame": reference_first_foot,
        "replay_first_foot_frame": first_own_foot_frame,
        "reference_frame86_ball_speed_mps": reference_speed,
        "replay_frame86_ball_speed_mps": ball_speed[86 - START],
        "frame86_speed_error_mps": speed_error,
        "reference_frame86_ball_pelvis_distance_m": reference_distance,
        "replay_frame86_ball_pelvis_distance_m": ball_pelvis_distance[86 - START],
        "frame86_distance_error_m": distance_error,
        "own_nonfoot_seen": own_nonfoot_seen,
        "all_robot_bodies_safe": all_safe,
        "maximum_executed_torque_error_nm": maximum_executed_torque_error_nm,
        "training_proxy_fidelity_passed": passed,
        "promotion_authorized": False,
    }
    report["report_hash"] = hash_json(report)
    (output_dir / "report.json").write_text(
        json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", required=True, type=Path)
    parser.add_argument("--captured", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    report = replay(**vars(parser.parse_args()))
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "report_hash",
                    "reference_first_foot_frame",
                    "replay_first_foot_frame",
                    "frame86_speed_error_mps",
                    "frame86_distance_error_m",
                    "maximum_executed_torque_error_nm",
                    "training_proxy_fidelity_passed",
                )
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
