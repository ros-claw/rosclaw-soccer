"""Reconstruct real 8-G1 contact frames and measure ball/foot/shin clearance.

Finite differences are diagnostics on saved measured states, not motor actions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from rosclaw_soccer.providers.g1.joint_contract import G1_DDS_JOINT_NAMES
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.training.continuous_match_residual_ppo import collection_fixture
from rosclaw_soccer.training.receiving_classroom import r1_contact_tap_receiving_configuration
from rosclaw_soccer.training.receiving_experiment import simulate_r0_receiving_course
from rosclaw_soccer.training.role_receiving_courses import ReceivingCourse
from rosclaw_soccer.world.multi_player import build_g1_multi_player_stadium_model

COURSES = (
    ReceivingCourse("red.finisher", 92801, 1.25, 0.08),
    ReceivingCourse("red.finisher", 92803, 1.5, -0.08),
)


def _distances(model: mujoco.MjModel, data: mujoco.MjData, side: str) -> tuple[float, float]:
    ball = model.geom("ball_geom").id
    shin = model.geom(f"red_finisher_{side}_shin").id
    foot = [model.geom(f"red_finisher_{side}_foot{i}_collision").id for i in range(1, 8)]
    return (
        float(mujoco.mj_geomDistance(model, data, ball, shin, 10.0, None)),
        min(float(mujoco.mj_geomDistance(model, data, ball, g, 10.0, None)) for g in foot),
    )


def _reconstruct(
    model: mujoco.MjModel, data: mujoco.MjData, trace: dict[str, Any], frame: int
) -> None:
    mujoco.mj_resetData(model, data)
    root = model.joint("red_finisher_floating_base_joint").qposadr[0]
    data.qpos[root : root + 7] = np.asarray(trace["red_finisher_pelvis_pose"])[frame]
    joints = np.asarray(trace["red_finisher_joint_position"])[frame]
    for i, name in enumerate(G1_DDS_JOINT_NAMES):
        address = model.joint(f"red_finisher_{name}").qposadr[0]
        data.qpos[address] = joints[i]
    ball_joint = model.jnt_qposadr[model.body("ball").jntadr[0]]
    data.qpos[ball_joint : ball_joint + 7] = np.asarray(trace["ball_pose"])[frame]
    mujoco.mj_forward(model, data)


def audit(asset_root: Path, policy: Path, output: Path) -> dict[str, Any]:
    if output.exists() or output.resolve().is_relative_to(Path(__file__).resolve().parents[1]):
        raise ValueError("new external diagnostic output required")
    root = Path(__file__).resolve().parents[1]
    sources = {
        name: hash_bytes((root / name).read_bytes())
        for name in (
            "scripts/rsi_r1_shin_clearance_v121.py",
            "src/rosclaw_soccer/training/receiving_experiment.py",
            "src/rosclaw_soccer/skills/team/independent_team_world.py",
            "src/rosclaw_soccer/world/multi_player.py",
        )
    }
    fixture = collection_fixture(asset_root, keeper_preview=True)
    world, _ = r1_contact_tap_receiving_configuration()
    model = build_g1_multi_player_stadium_model(
        asset_root,
        players=fixture.players,
        spec=fixture.goal,
        left_goal_plane_x_m=world.left_goal_plane_x_m if world.bilateral_goals else None,
    )
    data = mujoco.MjData(model)
    rows = []
    for course in COURSES:
        result, trace = simulate_r0_receiving_course(
            asset_root=asset_root,
            reference_policy_path=policy,
            course=course,
            scenario_id=f"s199.rsi.r1.shin-clearance.{course.seed}",
            configuration_profile="R1_CONTACT_TAP",
        )
        if not result.to_dict()["safe"]:
            raise ValueError("safe same-world diagnostic baseline required")
        nonfoot = np.asarray(trace["ball_nonfoot_contact_agent_code"])
        force = np.asarray(trace["ball_nonfoot_contact_force_n"])
        collision_frames = np.flatnonzero((nonfoot > 0) & (force > 0))
        collision_frames = collision_frames[(collision_frames >= 20) & (collision_frames < 120)]
        if len(collision_frames) != 1:
            raise ValueError("expected one measured same-player shin collision per course")
        collision_frame = int(collision_frames[0])
        geom_id = int(np.asarray(trace["ball_nonfoot_contact_geom_id"])[collision_frame])
        geom_name = model.geom(geom_id).name
        if geom_name not in ("red_finisher_left_shin", "red_finisher_right_shin"):
            raise ValueError("same model must identify the actual focal shin")
        side = "left" if "_left_" in geom_name else "right"
        timeline = []
        for frame in range(collision_frame - 5, collision_frame + 3):
            _reconstruct(model, data, trace, frame)
            shin_m, foot_m = _distances(model, data, side)
            timeline.append(
                {
                    "frame": frame,
                    "time_sec": float(np.asarray(trace["time"])[frame]),
                    "shin_clearance_m": shin_m,
                    "nearest_foot_clearance_m": foot_m,
                    "measured_nonfoot_force_n": float(force[frame]),
                }
            )
        probe_frame = collision_frame - 3
        _reconstruct(model, data, trace, probe_frame)
        base_shin, base_foot = _distances(model, data, side)
        gradient = []
        for name in G1_DDS_JOINT_NAMES[:12]:
            address = model.joint(f"red_finisher_{name}").qposadr[0]
            center = float(data.qpos[address])
            data.qpos[address] = center + 0.02
            mujoco.mj_forward(model, data)
            plus = _distances(model, data, side)
            data.qpos[address] = center - 0.02
            mujoco.mj_forward(model, data)
            minus = _distances(model, data, side)
            data.qpos[address] = center
            gradient.append(
                {
                    "joint": name,
                    "shin_m_per_rad": (plus[0] - minus[0]) / 0.04,
                    "foot_m_per_rad": (plus[1] - minus[1]) / 0.04,
                }
            )
        rows.append(
            {
                "course": vars(course),
                "result_hash": hash_json(result.to_dict()),
                "collision_frame": collision_frame,
                "collision_geom_id": geom_id,
                "collision_geom_name": geom_name,
                "side": side,
                "timeline": timeline,
                "probe_frame": probe_frame,
                "probe_shin_clearance_m": base_shin,
                "probe_foot_clearance_m": base_foot,
                "local_joint_sensitivity": gradient,
            }
        )
    report = {
        "schema": "rosclaw_soccer.rsi.r1_shin_clearance_v121.result.v1",
        "partition": "CONSUMED_EIGHT_G1_DIAGNOSTIC",
        "source_hashes": sources,
        "policy_hash": hash_bytes(policy.read_bytes()),
        "rollout_count": len(rows),
        "courses": rows,
        "status": "MEASURED_PRECONTACT_SHIN_GEOMETRY",
        "training_authorized": False,
        "promotion_authorized": False,
        "video_authorized": False,
        "activation_ceiling": "SIM_ONLY",
    }
    report["report_hash"] = hash_json(report)
    output.mkdir(parents=True)
    (output / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    if any(hash_bytes((root / name).read_bytes()) != digest for name, digest in sources.items()):
        raise ValueError("source drift during geometry diagnosis")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.asset_root, args.policy, args.output)
    print(json.dumps({"status": report["status"], "report_hash": report["report_hash"]}))


if __name__ == "__main__":
    main()
