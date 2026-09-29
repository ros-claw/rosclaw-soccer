"""Prove same-frame shin differential is accurate and read-only in eight-G1 physics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from rsi_r1_shin_clearance_v121 import _reconstruct
from rsi_r1_taskspace_feedback_v125 import COURSES

from rosclaw_soccer.providers.g1.joint_contract import G1_DDS_JOINT_NAMES
from rosclaw_soccer.rsi.receiving_shin_clearance_tap import ReceivingShinClearanceTap
from rosclaw_soccer.rsi.team_receive_contact_evidence import (
    ReceiveContactEvidence,
    ReceiveContactMailbox,
)
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.skills.team.shin_clearance import measure_team_shin_clearance
from rosclaw_soccer.training.continuous_match_residual_ppo import collection_fixture
from rosclaw_soccer.training.receiving_classroom import r1_contact_tap_receiving_configuration
from rosclaw_soccer.training.receiving_experiment import simulate_r0_receiving_course
from rosclaw_soccer.training.receiving_oracle_schedule import ReceivingOracleSchedule
from rosclaw_soccer.world.multi_player import build_g1_multi_player_stadium_model

SCHEDULE = ReceivingOracleSchedule("red.finisher", "A1_body29", 20, 10, ((0.0,) * 29,))
PHYSICAL_KEYS = (
    "ball_pose",
    "ball_velocity",
    "ball_contact_agent_code",
    "ball_contact_foot_code",
    "ball_nonfoot_contact_agent_code",
    "red_finisher_joint_position",
    "red_finisher_joint_velocity",
)


def audit(asset_root: Path, policy: Path, output: Path) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    if output.exists() or output.resolve().is_relative_to(root):
        raise ValueError("new external SIM_ONLY shin observation evidence required")
    sources = {
        name: hash_bytes((root / name).read_bytes())
        for name in (
            "scripts/rsi_r1_shin_feedback_tap_v133.py",
            "src/rosclaw_soccer/rsi/receiving_shin_clearance_tap.py",
            "src/rosclaw_soccer/skills/team/shin_clearance.py",
            "src/rosclaw_soccer/training/receiving_feedback.py",
            "src/rosclaw_soccer/skills/team/independent_team_world.py",
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
    dofs = tuple(
        tuple(
            model.joint(f"red_finisher_{name}").dofadr[0] for name in G1_DDS_JOINT_NAMES[i : i + 6]
        )
        for i in (0, 6)
    )
    qads = tuple(
        tuple(
            model.joint(f"red_finisher_{name}").qposadr[0] for name in G1_DDS_JOINT_NAMES[i : i + 6]
        )
        for i in (0, 6)
    )
    ball_geom_id = model.geom("ball_geom").id
    rows = []
    output.mkdir(parents=True)
    for course, side, frames in zip(COURSES, (0, 1), ((37, 38, 39), (23, 24, 25)), strict=True):
        parent, parent_trace = simulate_r0_receiving_course(
            asset_root=asset_root,
            reference_policy_path=policy,
            course=course,
            scenario_id=f"s199.rsi.r1.shin-tap.{course.seed}",
            configuration_profile="R1_CONTACT_TAP",
        )
        mailbox = ReceiveContactMailbox(course.agent_id)
        tap = ReceivingShinClearanceTap(course.agent_id, SCHEDULE.contract_hash, mailbox)
        child, child_trace = simulate_r0_receiving_course(
            asset_root=asset_root,
            reference_policy_path=policy,
            course=course,
            scenario_id=f"s199.rsi.r1.shin-tap.{course.seed}",
            configuration_profile="R1_CONTACT_TAP",
            oracle=SCHEDULE,
            feedback_provider=tap,
            physics_evidence_consumers={course.agent_id: ReceiveContactEvidence(mailbox)},
        )
        tape = tap.arrays()
        comparisons = {
            key: bool(np.array_equal(parent_trace[key], child_trace[key])) for key in PHYSICAL_KEYS
        }
        checks = []
        for frame in frames:
            _reconstruct(model, data, parent_trace, frame)
            sensor = measure_team_shin_clearance(
                model=model,
                data=data,
                agent_id=course.agent_id,
                frame=frame,
                ball_geom_id=ball_geom_id,
                leg_dof_ids=dofs,
            )
            grad = np.asarray(sensor.gradient_m_per_rad[side])
            finite = []
            for address in qads[side]:
                center = float(data.qpos[address])
                distances = []
                for direction in (1, -1):
                    data.qpos[address] = center + direction * 1e-4
                    mujoco.mj_forward(model, data)
                    distances.append(
                        float(
                            mujoco.mj_geomDistance(
                                model,
                                data,
                                model.geom(f"red_finisher_{('left', 'right')[side]}_shin").id,
                                ball_geom_id,
                                100.0,
                                None,
                            )
                        )
                    )
                data.qpos[address] = center
                finite.append((distances[0] - distances[1]) / 2e-4)
            check = {
                "frame": frame,
                "side": ("left", "right")[side],
                "clearance_m": sensor.clearance_m[side],
                "maximum_gradient_error_m_per_rad": float(np.max(np.abs(grad - finite))),
            }
            checks.append(check)
        row = {
            "course": vars(course),
            "parent_safe": parent.to_dict()["safe"],
            "child_safe": child.to_dict()["safe"],
            "physical_array_equal": comparisons,
            "physics_evidence_fault_agents": child.to_dict()["physics_evidence_fault_agents"],
            "tape_frames": len(tape["frame"]),
            "tape_shape": list(tape["shin_gradient_m_per_rad"].shape),
            "tape_minimum_clearance_m": float(np.min(tape["shin_clearance_m"])),
            "gradient_checks": checks,
            "parent_result_hash": hash_json(parent.to_dict()),
            "child_result_hash": hash_json(child.to_dict()),
        }
        np.savez_compressed(output / f"shin-tape-{course.seed}.npz", **tape)  # type: ignore[arg-type]
        row["tape_hash"] = hash_bytes((output / f"shin-tape-{course.seed}.npz").read_bytes())
        rows.append(row)
        (output / "progress.json").write_text(json.dumps(rows, indent=2) + "\n")
        print(
            json.dumps({"seed": course.seed, "equal": all(comparisons.values()), "checks": checks}),
            flush=True,
        )
    qualified = all(
        row["parent_safe"]
        and row["child_safe"]
        and not row["physics_evidence_fault_agents"]
        and all(row["physical_array_equal"].values())
        and row["tape_frames"] > 100
        and row["tape_shape"] == [row["tape_frames"], 2, 6]
        and all(c["maximum_gradient_error_m_per_rad"] < 0.005 for c in row["gradient_checks"])
        for row in rows
    )
    report = {
        "schema": "rosclaw_soccer.rsi.r1_shin_feedback_tap_v133.result.v1",
        "partition": "CONSUMED_EIGHT_G1_SHIN_OBSERVATION",
        "source_hashes": sources,
        "policy_hash": hash_bytes(policy.read_bytes()),
        "schedule_hash": SCHEDULE.contract_hash,
        "rollout_count": 4,
        "courses": rows,
        "status": "SHIN_DIFFERENTIAL_OBSERVATION_QUALIFIED"
        if qualified
        else "REJECTED_SHIN_OBSERVATION",
        "training_authorized": False,
        "promotion_authorized": False,
        "video_authorized": False,
        "activation_ceiling": "SIM_ONLY",
    }
    report["report_hash"] = hash_json(report)
    (output / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    if any(hash_bytes((root / name).read_bytes()) != digest for name, digest in sources.items()):
        raise ValueError("source drift during shin tap audit")
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
