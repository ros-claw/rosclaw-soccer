"""Consumed R1 three-player physical chain with a disjoint receiver motor.

This is a diagnostic integration bridge, not a promotion or video gate.
"""

from __future__ import annotations

import argparse
import json
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from rsi_r1_current_parent_replay import _Recorder
from rsi_team_taskspace_first_touch import TeamSwingMotor

from rosclaw_soccer.rsi.taskspace_swing_probe import recover_swing_joint_boundary
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.skills.team.independent_team_world import (
    IndependentTeamWorldScenario,
    simulate_independent_team_world,
)
from rosclaw_soccer.skills.team.motor_option import TeamMotorObservation, TeamMotorTarget
from rosclaw_soccer.skills.team.motor_retirement import TeamMotorRetirement
from rosclaw_soccer.training.continuous_competitive_match_growth import (
    build_continuous_competitive_fixture,
    default_continuous_match_config,
)
from rosclaw_soccer.training.pass_contact_chain import inspect_pass_contact_chain
from rosclaw_soccer.training.phase_conditioned_strike_growth import (
    default_phase_strike_controller,
    default_phase_strike_option,
    default_phase_strike_teacher,
)


class RetiringReceiverMotor(TeamSwingMotor):  # type: ignore[misc]
    """Yield only after observed own-foot contact, never by timeout."""

    def __init__(self, agent_id: str, enabled: bool, action: dict[str, Any]) -> None:
        super().__init__(agent_id, enabled, action)
        self.idle_frame: int | None = None

    def propose(self, observation: TeamMotorObservation) -> TeamMotorTarget | None:
        if self.first_contact_frame is not None:
            if observation.agent_id != self.agent_id or observation.frame != self.next_frame:
                raise ValueError("same-player sequential retirement required")
            self.next_frame += 1
            self.idle_frame = observation.frame
            return None
        return super().propose(observation)  # type: ignore[no-any-return]

    def retirement_request(self, *, frame: int, time_sec: float) -> TeamMotorRetirement | None:
        if self.first_contact_frame is None or self.idle_frame != frame:
            return None
        return TeamMotorRetirement(self.agent_id, frame, time_sec, self.contract_hash)


def select_grounded_receiver_foot(
    feet: np.ndarray[Any, Any], ball: np.ndarray[Any, Any], *, radius_m: float = 0.55
) -> int:
    """Select a planar-near foot for rolling passes from either direction."""
    if (
        feet.shape != (2, 3)
        or ball.shape != (3,)
        or radius_m not in (0.30, 0.40, 0.55)
        or not np.isfinite(feet).all()
        or not np.isfinite(ball).all()
    ):
        raise ValueError("finite two-foot and ball positions required")
    if ball[2] > 0.20:
        return -1
    candidates = (
        side
        for side in (0, 1)
        if 0.18 <= np.linalg.norm(ball[:2] - feet[side, :2]) <= radius_m
        and feet[side, 2] <= feet[1 - side, 2] + 0.02
    )
    return min(candidates, key=lambda side: np.linalg.norm(ball[:2] - feet[side, :2]), default=-1)


def rolling_receive_joint_delta(
    foot: np.ndarray[Any, Any],
    ball: np.ndarray[Any, Any],
    jacobian: np.ndarray[Any, Any],
    baseline: np.ndarray[Any, Any],
    limits: np.ndarray[Any, Any],
    *,
    cap_rad: float = 0.08,
    vertical_m: float = 0.04,
    radius_m: float = 0.55,
) -> np.ndarray[Any, Any]:
    """Small, direction-equivariant foot reach; no global +x assumption."""
    if (
        foot.shape != (3,)
        or ball.shape != (3,)
        or jacobian.shape != (3, 6)
        or baseline.shape != (6,)
        or limits.shape != (6, 2)
        or cap_rad not in (0.02, 0.04, 0.08)
        or vertical_m not in (0.0, 0.02, 0.04)
        or radius_m not in (0.30, 0.40, 0.55)
        or not all(np.isfinite(a).all() for a in (foot, ball, jacobian, baseline, limits))
        or np.any(limits[:, 0] >= limits[:, 1])
    ):
        raise ValueError("finite rolling receive kinematics and limits required")
    offset = ball[:2] - foot[:2]
    distance = float(np.linalg.norm(offset))
    if not 0.18 <= distance <= radius_m or ball[2] > 0.20:
        return np.zeros(6)
    desired = np.array([*(offset / distance * min(distance - 0.14, 0.08)), vertical_m])
    gram = jacobian @ jacobian.T + 0.05**2 * np.eye(3)
    delta = np.clip(jacobian.T @ np.linalg.solve(gram, desired), -cap_rad, cap_rad)
    proposed = baseline + delta
    safe = (
        (
            (baseline >= limits[:, 0])
            & (baseline <= limits[:, 1])
            & (proposed >= limits[:, 0])
            & (proposed <= limits[:, 1])
        )
        | ((baseline < limits[:, 0]) & (delta > 0))
        | ((baseline > limits[:, 1]) & (delta < 0))
    )
    return np.where(safe, delta, 0.0)


class GroundedRetiringReceiverMotor(RetiringReceiverMotor):
    """Bounded rolling-ball foot reach when the airborne selector is inactive."""

    def propose(self, observation: TeamMotorObservation) -> TeamMotorTarget | None:
        proposed = super().propose(observation)
        if (
            proposed is None
            or int(getattr(self, "side", -1)) >= 0
            or observation.frame < self.action["entry_frame"]
        ):
            return proposed
        kinematics = observation.foot_kinematics
        if kinematics is None or observation.foundation is None:
            raise ValueError("grounded receiver requires live foot and foundation context")
        feet = np.asarray(kinematics.foot_position_world_m, dtype=float)
        ball = np.asarray(observation.qpos[36:39], dtype=float)
        if self.action["rolling_requires_commitment"] and not observation.committed_receiver:
            return proposed
        side = select_grounded_receiver_foot(feet, ball, radius_m=self.action["rolling_radius_m"])
        if side < 0:
            return proposed
        self.side = side
        ids = slice(side * 6, side * 6 + 6)
        baseline = np.asarray(observation.foundation.target.target_rad, dtype=float)
        limits = np.asarray(kinematics.leg_joint_limits_rad, dtype=float)[side]
        jacobian = np.asarray(kinematics.foot_linear_jacobian_world, dtype=float)[side]
        delta = rolling_receive_joint_delta(
            feet[side],
            ball,
            jacobian,
            baseline[ids],
            limits,
            cap_rad=self.action["rolling_cap_rad"],
            vertical_m=self.action["rolling_vertical_m"],
            radius_m=self.action["rolling_radius_m"],
        )
        delta = recover_swing_joint_boundary(
            np.asarray(observation.qpos, dtype=float)[7 + side * 6 : 13 + side * 6],
            delta,
            limits,
            cap_rad=0.04,
        )
        target = baseline.copy()
        target[ids] += delta
        residual = target - baseline
        self.last_residual = residual.copy()
        self.observations["side"][-1] = np.asarray(side)
        self.observations["residual"][-1] = residual.copy()
        self.observations["executed"][-1] = target.copy()
        return TeamMotorTarget(tuple(float(x) for x in target), proposed.kp, proposed.kd)


def run(
    asset_root: Path,
    output_dir: Path,
    *,
    enabled: bool,
    motor_present: bool = True,
    grounded: bool = False,
    rolling_cap_rad: float = 0.08,
    rolling_radius_m: float = 0.55,
    rolling_vertical_m: float = 0.04,
    rolling_requires_commitment: bool = False,
    ankle_braking: float | None = None,
    retired_ankle_braking: float | None = None,
    ball_x_m: float = 1.92,
    ball_y_m: float = -0.80,
    seed: int = 207_200,
    stance_lateral_m: float = -0.17,
    pass_speed_mps: float = 0.80,
    preview_pass: bool = False,
) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    if (
        rolling_cap_rad not in (0.02, 0.04, 0.08)
        or rolling_radius_m not in (0.30, 0.40, 0.55)
        or rolling_vertical_m not in (0.0, 0.02, 0.04)
        or type(rolling_requires_commitment) is not bool
        or ankle_braking not in (None, 8.0, 12.0, 16.0)
        or retired_ankle_braking not in (None, 8.0, 12.0, 16.0)
        or ankle_braking is not None
        and retired_ankle_braking is not None
        or retired_ankle_braking is not None
        and not motor_present
        or not 1.80 <= ball_x_m <= 2.04
        or not -0.92 <= ball_y_m <= -0.68
        or type(seed) is not int
        or not 0 <= seed < 2**32
        or not -0.30 <= stance_lateral_m <= -0.05
        or not 0.60 <= pass_speed_mps <= 1.40
        or type(preview_pass) is not bool
    ):
        raise ValueError("bounded rolling receive curriculum required")
    if output_dir.exists() or output_dir.resolve().is_relative_to(root):
        raise ValueError("new external evidence directory required")
    names = (
        "scripts/rsi_r1_receiver_bridge_v71.py",
        "scripts/rsi_team_taskspace_first_touch.py",
        "src/rosclaw_soccer/skills/team/independent_team_world.py",
        "src/rosclaw_soccer/skills/team/motor_retirement.py",
        "src/rosclaw_soccer/growth/independent_agent_cell.py",
        "src/rosclaw_soccer/training/pass_contact_chain.py",
    )
    sources = {name: hash_bytes((root / name).read_bytes()) for name in names}
    torch.set_num_threads(1)
    fixture = build_continuous_competitive_fixture(asset_root)
    if preview_pass:
        fixture = replace(
            fixture,
            cells=tuple(
                replace(
                    cell,
                    tactical_profile=replace(cell.tactical_profile, anticipatory_contact=True),
                )
                if cell.agent_id == "red.playmaker"
                else cell
                for cell in fixture.cells
            ),
        )
    world = replace(
        default_continuous_match_config(),
        simulation_duration_sec=8.70,
        disjoint_motor_backends=motor_present,
        retire_completed_motors=motor_present,
        outward_ankle_roll_braking_damping=ankle_braking,
        retired_motor_option_ankle_braking_damping=retired_ankle_braking,
    )
    option = replace(
        default_phase_strike_option(),
        task_context_bound=True,
        per_player_options_enabled=True,
    )
    teacher = replace(
        default_phase_strike_teacher(),
        pass_strike_foot_speed_mps=pass_speed_mps,
        pass_stroke_duration_sec=0.0,
        preferred_foot="nearest",
    )
    phase = replace(default_phase_strike_controller(), target_stance_lateral_m=stance_lateral_m)
    scenario = IndependentTeamWorldScenario(
        scenario_id="s199.rsi.r1.receiver-bridge.consumed",
        ball_initial_position_m=(ball_x_m, ball_y_m, 0.115),
        ball_initial_velocity_mps=(0.0, 0.0, 0.0),
        seed=seed,
    )
    action = {
        "entry_frame": 30,
        "forward_cap_m": 0.16,
        "lateral_cap_m": 0.10,
        "vertical_offset_m": 0.04,
        "swing_foot_acquisition_gap_m": 0.55,
        "swing_acquisition_max_lateral_gap_m": 0.22,
        "revalidate_swing_side": True,
        "joint_boundary_recovery_cap_rad": 0.04,
        "rolling_cap_rad": rolling_cap_rad,
        "rolling_radius_m": rolling_radius_m,
        "rolling_vertical_m": rolling_vertical_m,
        "rolling_requires_commitment": rolling_requires_commitment,
    }
    if grounded and (not enabled or not motor_present):
        raise ValueError("grounded reach requires an enabled disjoint motor")
    motor = (GroundedRetiringReceiverMotor if grounded else RetiringReceiverMotor)(
        "red.finisher", enabled, action
    )
    protocol = {
        "schema": "rosclaw_soccer.rsi.r1_receiver_bridge_v71.protocol.v1",
        "partition": "CONSUMED_DEV",
        "activation_ceiling": "SIM_ONLY",
        "enabled": enabled,
        "grounded": grounded,
        "ankle_braking": ankle_braking,
        "retired_ankle_braking": retired_ankle_braking,
        "motor_present": motor_present,
        "scenario": asdict(scenario),
        "world_config_hash": world.config_hash,
        "option_config_hash": option.config_hash,
        "teacher_config_hash": teacher.config_hash,
        "stance_lateral_m": stance_lateral_m,
        "pass_speed_mps": pass_speed_mps,
        "preview_pass": preview_pass,
        "motor_contract_hash": motor.contract_hash,
        "source_hashes": sources,
        "promotion_authorized": False,
        "video_authorized": False,
    }
    output_dir.mkdir(parents=True)
    (output_dir / "protocol.json").write_text(
        json.dumps(protocol, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    recorder = _Recorder()
    with (
        (output_dir / "simulation.log").open("x", encoding="utf-8") as log,
        redirect_stdout(log),
        redirect_stderr(log),
    ):
        result, trace = simulate_independent_team_world(
            asset_root=asset_root,
            roster=fixture.roster,
            cells=fixture.cells,
            players=fixture.players,
            scenario=scenario,
            goal=fixture.goal,
            config=world,
            contact_teacher_config=teacher,
            option_bridge_config=option,
            strike_phase_config=phase,
            motor_options={"red.finisher": motor} if motor_present else None,
            physics_evidence_consumers={"red.playmaker": recorder},
        )
    if {name: hash_bytes((root / name).read_bytes()) for name in names} != sources:
        raise RuntimeError("source drift during physics run")
    trace_path = output_dir / "trace.npz"
    np.savez_compressed(trace_path, **trace)  # type: ignore[arg-type]
    ids = tuple(sorted(agent.agent_id for agent in fixture.roster.agents))
    request_frames = np.flatnonzero(
        (trace["pass_source_agent_code"] > 0) & (trace["pass_target_agent_code"] > 0)
    )
    chain = None
    receiver = None
    request_frame = None
    request_time_sec = None
    if len(request_frames) and recorder.rows:
        first = int(request_frames[0])
        request_frame = first
        # The team's decision is made at the START of a 20 ms control frame;
        # trace["time"] records the END of that frame after 500 Hz physics.
        request_time_sec = 0.0 if first == 0 else float(trace["time"][first - 1])
        receiver = ids[int(trace["pass_target_agent_code"][first]) - 1]
        chain = asdict(
            inspect_pass_contact_chain(
                tuple(recorder.rows),
                sender_id="red.playmaker",
                receiver_id=receiver,
                request_time_sec=request_time_sec,
                allow_initial_request_before_stream=first == 0,
            )
        )
    ball = trace["ball_pose"][:, :3]
    crossing = None
    for frame in range(len(ball) - 1):
        if ball[frame, 0] < fixture.goal.plane_x_m <= ball[frame + 1, 0]:
            alpha = (fixture.goal.plane_x_m - ball[frame, 0]) / (
                ball[frame + 1, 0] - ball[frame, 0]
            )
            y, z = ball[frame, 1:3] + alpha * (ball[frame + 1, 1:3] - ball[frame, 1:3])
            crossing = {
                "frame": frame,
                "y_m": float(y),
                "z_m": float(z),
                "inside_geometry": bool(
                    abs(y - fixture.goal.target_y_m)
                    <= fixture.goal.width_m / 2 - fixture.goal.ball_radius_m
                    and fixture.goal.ball_radius_m
                    <= z
                    <= fixture.goal.height_m - fixture.goal.ball_radius_m
                ),
            }
            break
    report = {
        "schema": "rosclaw_soccer.rsi.r1_receiver_bridge_v71.report.v1",
        "protocol_hash": hash_bytes((output_dir / "protocol.json").read_bytes()),
        "trace_hash": hash_bytes(trace_path.read_bytes()),
        "result": result.to_dict(),
        "receiver": receiver,
        "request_frame": request_frame,
        "request_time_sec": request_time_sec,
        "chain": chain,
        "crossing": crossing,
        "chain_success": bool(
            result.safe
            and chain is not None
            and chain["clean_transfer_observed"]
            and crossing is not None
            and crossing["inside_geometry"]
            and chain["receiver_contact_sec"] is not None
            and chain["receiver_contact_sec"] < float(trace["time"][int(crossing["frame"])])
        ),
        "motor_active_frames": (
            int(np.count_nonzero(np.any(np.asarray(motor.observations["residual"]) != 0, axis=1)))
            if motor_present
            else 0
        ),
        "motor_first_foot_contact_frame": motor.first_contact_frame,
        "motor_peak_own_foot_force_n": max(motor.own_foot_force_peak_n, default=0.0),
        "promotion_authorized": False,
        "video_authorized": False,
    }
    report["report_hash"] = hash_json(report)
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--enabled", action="store_true")
    parser.add_argument("--grounded", action="store_true")
    parser.add_argument("--rolling-cap", type=float, choices=(0.02, 0.04, 0.08), default=0.08)
    parser.add_argument("--rolling-radius", type=float, choices=(0.30, 0.40, 0.55), default=0.55)
    parser.add_argument("--rolling-vertical", type=float, choices=(0.0, 0.02, 0.04), default=0.04)
    parser.add_argument("--rolling-requires-commitment", action="store_true")
    parser.add_argument("--ankle-braking", type=float, choices=(8.0, 12.0, 16.0))
    parser.add_argument("--retired-ankle-braking", type=float, choices=(8.0, 12.0, 16.0))
    parser.add_argument("--ball-x", type=float, default=1.92)
    parser.add_argument("--ball-y", type=float, default=-0.80)
    parser.add_argument("--seed", type=int, default=207_200)
    parser.add_argument("--stance", type=float, default=-0.17)
    parser.add_argument("--pass-speed", type=float, default=0.80)
    parser.add_argument("--preview-pass", action="store_true")
    parser.add_argument("--no-motor", action="store_true")
    args = parser.parse_args()
    report = run(
        args.asset_root,
        args.output_dir,
        enabled=args.enabled,
        motor_present=not args.no_motor,
        grounded=args.grounded,
        rolling_cap_rad=args.rolling_cap,
        rolling_radius_m=args.rolling_radius,
        rolling_vertical_m=args.rolling_vertical,
        rolling_requires_commitment=args.rolling_requires_commitment,
        ankle_braking=args.ankle_braking,
        retired_ankle_braking=args.retired_ankle_braking,
        ball_x_m=args.ball_x,
        ball_y_m=args.ball_y,
        seed=args.seed,
        stance_lateral_m=args.stance,
        pass_speed_mps=args.pass_speed,
        preview_pass=args.preview_pass,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
