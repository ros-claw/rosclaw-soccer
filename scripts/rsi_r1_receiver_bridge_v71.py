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

from rosclaw_soccer.rsi.b6_microphysics_observer import B6MicrophysicsObserver
from rosclaw_soccer.rsi.b6_velocity_cushion import velocity_match_joint_delta
from rosclaw_soccer.rsi.taskspace_swing_probe import recover_swing_joint_boundary
from rosclaw_soccer.rsi.team_receive_body_tap import TeamReceiveBodyTap
from rosclaw_soccer.rsi.team_receive_contact_evidence import (
    ReceiveContactEvidence,
    ReceiveContactMailbox,
)
from rosclaw_soccer.rsi.team_receive_contact_phase_actor import TeamReceiveContactPhaseActor
from rosclaw_soccer.rsi.team_receive_phase_actor import TeamReceivePhaseActor
from rosclaw_soccer.rsi.team_strike_feedback_navigation import TeamStrikeFeedbackNavigation
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.skills.team.independent_team_world import (
    IndependentTeamWorldScenario,
    simulate_independent_team_world,
)
from rosclaw_soccer.skills.team.motor_option import TeamMotorObservation, TeamMotorTarget
from rosclaw_soccer.skills.team.motor_retirement import TeamMotorRetirement
from rosclaw_soccer.skills.team.navigation_option import TeamNavigationPolicy
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


class RetiringReceiverMotor(TeamSwingMotor):
    """Yield only after observed own-foot contact, never by timeout."""

    def __init__(self, agent_id: str, enabled: bool, action: dict[str, Any]) -> None:
        super().__init__(agent_id, enabled, action)
        self.idle_frame: int | None = None

    def propose(self, observation: TeamMotorObservation) -> TeamMotorTarget | None:  # type: ignore[override]
        if self.first_contact_frame is not None:
            if observation.agent_id != self.agent_id or observation.frame != self.next_frame:
                raise ValueError("same-player sequential retirement required")
            self.next_frame += 1
            self.idle_frame = observation.frame
            return None
        return super().propose(observation)

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

    def propose(self, observation: TeamMotorObservation) -> TeamMotorTarget | None:  # type: ignore[override]
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


class VelocityCushionReceiverMotor(GroundedRetiringReceiverMotor):
    """SIM_ONLY bounded contact-preparation correction on measured player state."""

    def propose(self, observation: TeamMotorObservation) -> TeamMotorTarget | None:  # type: ignore[override]
        proposed = super().propose(observation)
        if proposed is None or not observation.committed_receiver:
            return proposed
        kinematics = observation.foot_kinematics
        if (
            kinematics is None
            or kinematics.foot_linear_velocity_world_mps is None
            or observation.foundation is None
        ):
            raise ValueError("velocity cushion requires live same-frame kinematics and foundation")
        ball_velocity = np.asarray(observation.qvel[35:38], dtype=float)
        if float(np.linalg.norm(ball_velocity[:2])) < 0.30:
            return proposed
        ball = np.asarray(observation.qpos[36:39], dtype=float)
        feet = np.asarray(kinematics.foot_position_world_m, dtype=float)
        side = select_grounded_receiver_foot(feet, ball, radius_m=0.55)
        if side < 0:
            return proposed
        ids = slice(side * 6, side * 6 + 6)
        baseline = np.asarray(observation.foundation.target.target_rad, dtype=float)
        target = np.asarray(proposed.target_rad, dtype=float)
        limits = np.asarray(kinematics.leg_joint_limits_rad, dtype=float)[side]
        delta = velocity_match_joint_delta(
            jacobian=np.asarray(kinematics.foot_linear_jacobian_world, dtype=float)[side],
            foot_velocity_mps=np.asarray(kinematics.foot_linear_velocity_world_mps, dtype=float)[
                side
            ],
            ball_velocity_mps=ball_velocity,
            gain=self.action["velocity_cushion_gain"],
        )
        target[ids] = np.clip(target[ids] + delta, limits[:, 0], limits[:, 1])
        residual = target - baseline
        self.last_residual = residual.copy()
        self.observations["side"][-1] = np.asarray(side)
        self.observations["residual"][-1] = residual.copy()
        self.observations["executed"][-1] = target.copy()
        return TeamMotorTarget(tuple(float(x) for x in target), proposed.kp, proposed.kd)


def directed_pass_joint_delta(
    *,
    jacobian: np.ndarray[Any, Any],
    measured_velocity_mps: np.ndarray[Any, Any],
    ball_xyz: np.ndarray[Any, Any],
    receiver_target_xyz: np.ndarray[Any, Any],
    speed_mps: float,
) -> np.ndarray[Any, Any]:
    """Damped, causal foot-velocity proposal toward the visible teammate target."""
    if (
        jacobian.shape != (3, 6)
        or measured_velocity_mps.shape != (3,)
        or ball_xyz.shape != (3,)
        or receiver_target_xyz.shape != (3,)
        or speed_mps not in (1.0, 1.5, 2.0)
        or not all(
            np.isfinite(a).all()
            for a in (jacobian, measured_velocity_mps, ball_xyz, receiver_target_xyz)
        )
    ):
        raise ValueError("finite bounded directed pass context required")
    offset = receiver_target_xyz[:2] - ball_xyz[:2]
    distance = float(np.linalg.norm(offset))
    if distance < 0.30:
        return np.zeros(6)
    desired = np.array([*(speed_mps * offset / distance), 0.0])
    gram = jacobian @ jacobian.T + 0.05**2 * np.eye(3)
    velocity_error = desired - measured_velocity_mps
    return np.clip(0.08 * jacobian.T @ np.linalg.solve(gram, velocity_error), -0.20, 0.20)


def biased_pass_target(
    ball_xyz: np.ndarray[Any, Any],
    receiver_xyz: np.ndarray[Any, Any],
    lateral_bias_m: float,
) -> np.ndarray[Any, Any]:
    """Move a visible teammate target along the pass-orthogonal axis only."""
    if (
        ball_xyz.shape != (3,)
        or receiver_xyz.shape != (3,)
        or lateral_bias_m not in (-0.40, -0.20, 0.0, 0.20, 0.40)
        or not np.isfinite(ball_xyz).all()
        or not np.isfinite(receiver_xyz).all()
    ):
        raise ValueError("finite bounded pass target required")
    axis = receiver_xyz[:2] - ball_xyz[:2]
    distance = float(np.linalg.norm(axis))
    if distance < 0.30:
        return receiver_xyz.copy()
    axis /= distance
    shifted = receiver_xyz.copy()
    shifted[:2] += lateral_bias_m * np.array([-axis[1], axis[0]])
    return shifted


class DirectedRetiringPassMotor(RetiringReceiverMotor):
    """A bounded velocity-directed teacher candidate, not a trained policy."""

    def propose(self, observation: TeamMotorObservation) -> TeamMotorTarget | None:  # type: ignore[override]
        proposed = super().propose(observation)
        if (
            proposed is None
            or int(getattr(self, "side", -1)) < 0
            or self.first_contact_frame is not None
            or observation.intent != "pass"
        ):
            return proposed
        kinematics = observation.foot_kinematics
        if kinematics is None or kinematics.foot_linear_velocity_world_mps is None:
            raise ValueError("directed pass requires live same-frame foot velocity")
        side = int(self.side)
        ids = slice(side * 6, side * 6 + 6)
        jacobian = np.asarray(kinematics.foot_linear_jacobian_world, dtype=float)[side]
        measured = np.asarray(kinematics.foot_linear_velocity_world_mps, dtype=float)[side]
        q = np.asarray(observation.qpos, dtype=float)
        target = np.asarray(proposed.target_rad, dtype=float)
        baseline = np.asarray(observation.foundation.target.target_rad, dtype=float)  # type: ignore[union-attr]
        velocity_delta = directed_pass_joint_delta(
            jacobian=jacobian,
            measured_velocity_mps=measured,
            ball_xyz=q[36:39],
            receiver_target_xyz=biased_pass_target(
                q[36:39],
                np.asarray(observation.target_position_m, dtype=float),
                self.action["pass_lateral_bias_m"],
            ),
            speed_mps=self.action["directed_pass_speed_mps"],
        )
        delta = np.clip(target[ids] - baseline[ids] + velocity_delta, -0.35, 0.35)
        limits = np.asarray(kinematics.leg_joint_limits_rad, dtype=float)[side]
        delta = recover_swing_joint_boundary(
            q[7 + side * 6 : 13 + side * 6], delta, limits, cap_rad=0.04
        )
        target[ids] = np.clip(baseline[ids] + delta, limits[:, 0], limits[:, 1])
        residual = target - baseline
        self.last_residual = residual.copy()
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
    option_ankle_braking: float | None = None,
    option_joint_guard_margin_rad: float | None = None,
    retired_ankle_braking: float | None = None,
    ball_x_m: float = 1.92,
    ball_y_m: float = -0.80,
    seed: int = 207_200,
    stance_lateral_m: float = -0.17,
    pass_speed_mps: float = 0.80,
    preview_pass: bool = False,
    duration_sec: float = 8.70,
    precontact_pass_standoff_m: float | None = None,
    motor_agent_id: str = "red.finisher",
    motor_entry_frame: int = 30,
    strike_through_m: float = 0.0,
    directed_pass_speed_mps: float = 0.0,
    pass_lateral_bias_m: float = 0.0,
    dual_receiver_motor: bool = False,
    velocity_cushion_gain: float = 0.0,
    receive_velocity_prediction_sec: float = 0.0,
    receive_velocity_prediction_lateral_only: bool = False,
    handoff_profile: str = "legacy",
    receive_profile: str = "legacy",
    phase_profile: str = "default",
    navigation_profile: str = "none",
    navigation_phase_weights: tuple[float, float, float, float, float] | None = None,
    navigation_post_weights: tuple[float, float, float, float, float] | None = None,
    teacher_profile: str = "default",
    receive_teacher_profile: str = "default",
    capture_profile: str = "none",
    receive_teacher_tuning: tuple[float, float] | None = None,
    capture_b6_microphysics: bool = False,
) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    if (
        rolling_cap_rad not in (0.02, 0.04, 0.08)
        or rolling_radius_m not in (0.30, 0.40, 0.55)
        or rolling_vertical_m not in (0.0, 0.02, 0.04)
        or type(rolling_requires_commitment) is not bool
        or ankle_braking not in (None, 8.0, 12.0, 16.0)
        or option_ankle_braking not in (None, 8.0, 12.0, 16.0)
        or option_joint_guard_margin_rad not in (None, 0.06, 0.08, 0.10)
        or ankle_braking is not None
        and option_ankle_braking is not None
        or retired_ankle_braking not in (None, 8.0, 12.0, 16.0)
        or ankle_braking is not None
        and retired_ankle_braking is not None
        or retired_ankle_braking is not None
        and not motor_present
        or not 1.80 <= ball_x_m <= 2.50
        or not -0.92 <= ball_y_m <= -0.68
        or type(seed) is not int
        or not 0 <= seed < 2**32
        or not -0.30 <= stance_lateral_m <= -0.05
        or not 0.60 <= pass_speed_mps <= 1.40
        or type(preview_pass) is not bool
        or duration_sec not in (8.70, 10.0)
        or precontact_pass_standoff_m not in (None, 0.25, 0.35, 0.45)
        or precontact_pass_standoff_m is not None
        and not preview_pass
        or motor_agent_id not in ("red.finisher", "red.playmaker")
        or motor_entry_frame not in (0, 5, 10, 15, 20, 25, 30)
        or strike_through_m not in (0.0, 0.08, 0.16)
        or directed_pass_speed_mps not in (0.0, 1.0, 1.5, 2.0)
        or pass_lateral_bias_m not in (-0.40, -0.20, 0.0, 0.20, 0.40)
        or pass_lateral_bias_m != 0.0
        and directed_pass_speed_mps == 0.0
        or directed_pass_speed_mps > 0
        and (not enabled or motor_agent_id != "red.playmaker")
        or type(dual_receiver_motor) is not bool
        or velocity_cushion_gain not in (0.0, 0.5, 1.0, 2.0)
        or velocity_cushion_gain > 0.0
        and not dual_receiver_motor
        or receive_velocity_prediction_sec not in (0.0, 0.04, 0.08, 0.12)
        or receive_velocity_prediction_sec > 0.0
        and dual_receiver_motor
        or type(receive_velocity_prediction_lateral_only) is not bool
        or receive_velocity_prediction_lateral_only
        and receive_velocity_prediction_sec == 0.0
        or dual_receiver_motor
        and (not motor_present or directed_pass_speed_mps == 0.0)
        or handoff_profile not in ("legacy", "strict", "tracking", "committed")
        or receive_profile not in ("legacy", "lateral", "retention", "lateral_retention")
        or receive_profile != "legacy"
        and handoff_profile == "legacy"
        or phase_profile not in ("default", "fast", "predictive", "ultra")
        or navigation_profile
        not in (
            "none",
            "receive_tap",
            "receive_phase",
            "receive_contact",
            "follow",
            "lead",
            "damped",
            "wide",
            "lease1",
            "lease2",
            "lease3",
            "lease2_damped",
        )
        or navigation_profile in ("receive_phase", "receive_contact")
        and (
            type(navigation_phase_weights) is not tuple
            or len(navigation_phase_weights) != 5
            or any(
                type(value) is not float or not np.isfinite(value) or abs(value) > 2.0
                for value in navigation_phase_weights
            )
        )
        or navigation_profile not in ("receive_phase", "receive_contact")
        and navigation_phase_weights is not None
        or navigation_profile == "receive_contact"
        and (
            type(navigation_post_weights) is not tuple
            or len(navigation_post_weights) != 5
            or any(
                type(value) is not float or not np.isfinite(value) or abs(value) > 2.0
                for value in navigation_post_weights
            )
        )
        or navigation_profile != "receive_contact"
        and navigation_post_weights is not None
        or navigation_profile == "receive_contact"
        and capture_b6_microphysics
        or teacher_profile not in ("default", "live_after_receive")
        or receive_teacher_profile not in ("default", "neutral", "soft", "cushion", "combined")
        or capture_profile not in ("none", "short", "medium", "long")
        or type(capture_b6_microphysics) is not bool
        or receive_teacher_tuning is not None
        and (
            type(receive_teacher_tuning) is not tuple
            or len(receive_teacher_tuning) != 2
            or not all(
                type(value) in (int, float) and np.isfinite(value)
                for value in receive_teacher_tuning
            )
            or not -0.30 <= receive_teacher_tuning[0] <= 0.30
            or not 0.12 <= receive_teacher_tuning[1] <= 0.24
            or receive_teacher_profile != "default"
        )
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
        "src/rosclaw_soccer/rsi/team_strike_feedback_navigation.py",
        "src/rosclaw_soccer/rsi/b6_microphysics_observer.py",
        "src/rosclaw_soccer/rsi/b6_velocity_cushion.py",
        "src/rosclaw_soccer/rsi/team_receive_body_tap.py",
        "src/rosclaw_soccer/rsi/team_receive_phase_actor.py",
        "src/rosclaw_soccer/rsi/team_receive_contact_phase_actor.py",
        "src/rosclaw_soccer/rsi/team_receive_contact_evidence.py",
        "src/rosclaw_soccer/skills/team/navigation_option.py",
        "src/rosclaw_soccer/growth/locomotion_contact_teacher.py",
        "src/rosclaw_soccer/skills/team/physics_evidence.py",
    )
    sources = {name: hash_bytes((root / name).read_bytes()) for name in names}
    torch.set_num_threads(1)
    fixture = build_continuous_competitive_fixture(asset_root)
    navigation: TeamNavigationPolicy | None = None
    contact_evidence: ReceiveContactEvidence | None = None
    if navigation_profile != "none":
        navigation_settings = {
            "follow": (0.8, 0.40, 0.36, -0.19, 0.0, 0.0),
            "lead": (1.2, 0.80, 0.36, -0.19, 0.0, 0.0),
            "damped": (1.2, 0.60, 0.36, -0.19, 0.3, 0.0),
            "wide": (1.2, 0.80, 0.50, -0.25, 0.2, 0.0),
            "lease1": (1.2, 0.60, 0.36, -0.19, 0.3, 1.0),
            "lease2": (1.2, 0.60, 0.36, -0.19, 0.3, 2.0),
            "lease3": (1.2, 0.60, 0.36, -0.19, 0.3, 3.0),
            "lease2_damped": (0.8, 0.40, 0.50, -0.25, 0.6, 2.0),
        }
        policy_path = asset_root / "policy/loco_mode/model/policy_29dof.pt"
        config_path = asset_root / "policy/loco_mode/config/LocoMode.yaml"
        if navigation_profile == "receive_tap":
            navigation = TeamReceiveBodyTap(
                agent_id="red.finisher",
                foundation_hash=hash_bytes(policy_path.read_bytes()),
                foundation_config_hash=hash_bytes(config_path.read_bytes()),
            )
        elif navigation_profile == "receive_phase":
            assert navigation_phase_weights is not None
            navigation = TeamReceivePhaseActor(
                agent_id="red.finisher",
                foundation_hash=hash_bytes(policy_path.read_bytes()),
                foundation_config_hash=hash_bytes(config_path.read_bytes()),
                weights=navigation_phase_weights,
            )
        elif navigation_profile == "receive_contact":
            assert navigation_phase_weights is not None and navigation_post_weights is not None
            mailbox = ReceiveContactMailbox("red.finisher")
            contact_evidence = ReceiveContactEvidence(mailbox)
            navigation = TeamReceiveContactPhaseActor(
                agent_id="red.finisher",
                foundation_hash=hash_bytes(policy_path.read_bytes()),
                foundation_config_hash=hash_bytes(config_path.read_bytes()),
                weights=navigation_phase_weights,
                post_weights=navigation_post_weights,
                mailbox=mailbox,
            )
        else:
            gain, horizon, depth, lateral, damping, commitment = navigation_settings[
                navigation_profile
            ]
            navigation = TeamStrikeFeedbackNavigation(
                agent_id="red.finisher",
                foundation_hash=hash_bytes(policy_path.read_bytes()),
                foundation_config_hash=hash_bytes(config_path.read_bytes()),
                position_gain=gain,
                prediction_horizon_sec=horizon,
                stance_depth_m=depth,
                stance_lateral_m=lateral,
                body_velocity_damping=damping,
                shot_commitment_sec=commitment,
            )
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
        simulation_duration_sec=duration_sec,
        disjoint_motor_backends=motor_present,
        retire_completed_motors=motor_present,
        outward_ankle_roll_braking_damping=ankle_braking,
        option_ankle_roll_braking_damping=option_ankle_braking,
        option_joint_guard_margin_rad=option_joint_guard_margin_rad,
        retired_motor_option_ankle_braking_damping=retired_ankle_braking,
        precontact_pass_standoff_m=precontact_pass_standoff_m,
        strict_receive_handoff=handoff_profile != "legacy",
        receiver_commitment_priority=handoff_profile in ("tracking", "committed"),
        preserve_launched_handoff=handoff_profile in ("tracking", "committed"),
        pass_target_commitment=handoff_profile == "committed",
        receive_lateral_braking=receive_profile in ("lateral", "lateral_retention"),
        controlled_possession_retention=receive_profile in ("retention", "lateral_retention"),
        post_receive_contact_control=capture_profile != "none",
        post_receive_hold_sec={"none": 0.20, "short": 0.20, "medium": 0.40, "long": 0.60}[
            capture_profile
        ],
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
        one_touch_finish_enabled=teacher_profile == "default",
    )
    if receive_teacher_profile != "default":
        teacher_settings = {
            "neutral": (0.0, 0.35, -0.06, 0.06),
            "soft": (-1.0, 0.0, -0.06, 0.06),
            "cushion": (-1.0, 0.35, -0.12, 0.02),
            "combined": (0.0, 0.0, -0.12, 0.02),
        }
        yaw, follow_speed, depth, minimum_forward = teacher_settings[receive_teacher_profile]
        teacher = replace(
            teacher,
            committed_receive_aim_yaw_bias_rad=yaw,
            committed_receive_follow_through_speed_mps=follow_speed,
            receive_cushion_depth_m=depth,
            receive_minimum_forward_target_m=minimum_forward,
        )
    if receive_teacher_tuning is not None:
        yaw, lateral_offset = receive_teacher_tuning
        teacher = replace(
            teacher,
            committed_receive_aim_yaw_bias_rad=yaw,
            committed_receive_ankle_lateral_offset_m=lateral_offset,
            committed_receive_follow_through_speed_mps=0.0,
            receive_cushion_depth_m=-0.12,
            receive_minimum_forward_target_m=0.02,
        )
    if receive_velocity_prediction_sec > 0.0:
        teacher = replace(
            teacher,
            receive_velocity_prediction_sec=receive_velocity_prediction_sec,
            receive_velocity_prediction_lateral_only=receive_velocity_prediction_lateral_only,
        )
    phase = replace(default_phase_strike_controller(), target_stance_lateral_m=stance_lateral_m)
    if phase_profile != "default":
        phase_settings = {
            "fast": (0.14, 0.60, 1.20),
            "predictive": (0.22, 1.00, 1.50),
            "ultra": (0.08, 0.80, 1.50),
        }
        capture, horizon, lateral_gain = phase_settings[phase_profile]
        phase = replace(
            phase,
            capture_duration_sec=capture,
            strike_contact_horizon_sec=horizon,
            orient_lateral_gain_per_sec=lateral_gain,
        )
    scenario = IndependentTeamWorldScenario(
        scenario_id="s199.rsi.r1.receiver-bridge.consumed",
        ball_initial_position_m=(ball_x_m, ball_y_m, 0.115),
        ball_initial_velocity_mps=(0.0, 0.0, 0.0),
        seed=seed,
    )
    action = {
        "entry_frame": motor_entry_frame,
        "forward_cap_m": 0.16,
        "lateral_cap_m": 0.10,
        "vertical_offset_m": 0.04,
        "swing_foot_acquisition_gap_m": 0.55,
        "swing_acquisition_max_lateral_gap_m": 0.22,
        "revalidate_swing_side": True,
        "joint_boundary_recovery_cap_rad": 0.04,
        "strike_through_m": strike_through_m,
        "directed_pass_speed_mps": directed_pass_speed_mps,
        "pass_lateral_bias_m": pass_lateral_bias_m,
        "rolling_cap_rad": rolling_cap_rad,
        "rolling_radius_m": rolling_radius_m,
        "rolling_vertical_m": rolling_vertical_m,
        "rolling_requires_commitment": rolling_requires_commitment,
    }
    if grounded and (not enabled or not motor_present):
        raise ValueError("grounded reach requires an enabled disjoint motor")
    motor_type = (
        DirectedRetiringPassMotor
        if directed_pass_speed_mps > 0
        else GroundedRetiringReceiverMotor
        if grounded
        else RetiringReceiverMotor
    )
    motor = motor_type(motor_agent_id, enabled, action)
    receiver_motor = None
    if dual_receiver_motor:
        receiver_action = {
            **action,
            "entry_frame": 30,
            "strike_through_m": 0.0,
            "directed_pass_speed_mps": 0.0,
            "rolling_cap_rad": 0.04,
            "rolling_radius_m": 0.30,
            "rolling_vertical_m": 0.02,
            "rolling_requires_commitment": True,
            "velocity_cushion_gain": velocity_cushion_gain,
        }
        receiver_type = (
            VelocityCushionReceiverMotor
            if velocity_cushion_gain > 0.0
            else GroundedRetiringReceiverMotor
        )
        receiver_motor = receiver_type("red.finisher", True, receiver_action)
    protocol = {
        "schema": "rosclaw_soccer.rsi.r1_receiver_bridge_v71.protocol.v1",
        "partition": "CONSUMED_DEV",
        "activation_ceiling": "SIM_ONLY",
        "enabled": enabled,
        "grounded": grounded,
        "ankle_braking": ankle_braking,
        "option_ankle_braking": option_ankle_braking,
        "option_joint_guard_margin_rad": option_joint_guard_margin_rad,
        "retired_ankle_braking": retired_ankle_braking,
        "motor_present": motor_present,
        "motor_agent_id": motor_agent_id,
        "motor_entry_frame": motor_entry_frame,
        "strike_through_m": strike_through_m,
        "directed_pass_speed_mps": directed_pass_speed_mps,
        "pass_lateral_bias_m": pass_lateral_bias_m,
        "scenario": asdict(scenario),
        "world_config_hash": world.config_hash,
        "option_config_hash": option.config_hash,
        "teacher_config_hash": teacher.config_hash,
        "stance_lateral_m": stance_lateral_m,
        "pass_speed_mps": pass_speed_mps,
        "preview_pass": preview_pass,
        "duration_sec": duration_sec,
        "precontact_pass_standoff_m": precontact_pass_standoff_m,
        "motor_contract_hash": motor.contract_hash,
        "dual_receiver_motor": dual_receiver_motor,
        "velocity_cushion_gain": velocity_cushion_gain,
        "receive_velocity_prediction_sec": receive_velocity_prediction_sec,
        "receive_velocity_prediction_lateral_only": receive_velocity_prediction_lateral_only,
        "handoff_profile": handoff_profile,
        "receive_profile": receive_profile,
        "phase_profile": phase_profile,
        "phase_config_hash": phase.config_hash,
        "navigation_profile": navigation_profile,
        "navigation_phase_weights": navigation_phase_weights,
        "navigation_post_weights": navigation_post_weights,
        "navigation_contract_hash": None if navigation is None else navigation.contract_hash,
        "teacher_profile": teacher_profile,
        "receive_teacher_profile": receive_teacher_profile,
        "receive_teacher_tuning": receive_teacher_tuning,
        "capture_profile": capture_profile,
        "capture_b6_microphysics": capture_b6_microphysics,
        "receiver_motor_contract_hash": (
            None if receiver_motor is None else receiver_motor.contract_hash
        ),
        "source_hashes": sources,
        "promotion_authorized": False,
        "video_authorized": False,
    }
    output_dir.mkdir(parents=True)
    (output_dir / "protocol.json").write_text(
        json.dumps(protocol, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    recorder = _Recorder()
    microphysics = B6MicrophysicsObserver("red.finisher") if capture_b6_microphysics else None
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
            motor_options=(
                {motor_agent_id: motor, "red.finisher": receiver_motor}
                if receiver_motor is not None
                else {motor_agent_id: motor}
                if motor_present
                else None
            ),
            navigation_policies=(None if navigation is None else {navigation.agent_id: navigation}),
            physics_evidence_consumers=(
                {"red.playmaker": recorder, "red.finisher": contact_evidence}
                if contact_evidence is not None
                else {"red.playmaker": recorder, "red.finisher": microphysics}
                if microphysics is not None
                else {"red.playmaker": recorder}
            ),
        )
    if {name: hash_bytes((root / name).read_bytes()) for name in names} != sources:
        raise RuntimeError("source drift during physics run")
    trace_path = output_dir / "trace.npz"
    np.savez_compressed(trace_path, **trace)  # type: ignore[arg-type]
    tap_context = None
    if isinstance(navigation, TeamReceiveBodyTap):
        tap_path = output_dir / "receive-body-context.npz"
        tap_arrays = navigation.arrays()
        np.savez_compressed(tap_path, **tap_arrays)  # type: ignore[arg-type]
        tap_context = {
            "archive_hash": hash_bytes(tap_path.read_bytes()),
            "sample_count": len(tap_arrays["frame"]),
            "contract_hash": navigation.contract_hash,
        }
    microphysics_report = None
    if microphysics is not None:
        microphysics_report = {
            "observer_contract_hash": microphysics.contract_hash,
            "complete": microphysics.complete,
            "observer_fault": "red.finisher" in result.physics_evidence_fault_agents,
            "first_foot_time_sec": microphysics.first_foot_time_sec,
            "incoming_ball_speed_mps": microphysics.incoming_ball_speed_mps,
            "archive_hash": None,
            "sample_count": 0,
        }
        if microphysics.complete:
            microphysics_arrays = microphysics.arrays()
            microphysics_path = output_dir / "b6-microphysics.npz"
            np.savez_compressed(microphysics_path, **microphysics_arrays)  # type: ignore[arg-type]
            microphysics_report["archive_hash"] = hash_bytes(microphysics_path.read_bytes())
            microphysics_report["sample_count"] = len(microphysics_arrays["time_sec"])
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
        "motor_agent_id": motor_agent_id if motor_present else None,
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
        "receiver_motor_active_frames": (
            0
            if receiver_motor is None
            else int(
                np.count_nonzero(
                    np.any(np.asarray(receiver_motor.observations["residual"]) != 0, axis=1)
                )
            )
        ),
        "receiver_motor_first_foot_contact_frame": (
            None if receiver_motor is None else receiver_motor.first_contact_frame
        ),
        "b6_microphysics": microphysics_report,
        "receive_body_context": tap_context,
        "receive_phase_actor": (
            {
                "active_frames": navigation.active_frames,
                "peak_delta_mps": navigation.peak_delta_mps,
                "contract_hash": navigation.contract_hash,
            }
            if isinstance(navigation, TeamReceivePhaseActor)
            else None
        ),
        "receive_contact_actor": (
            {
                "first_own_foot_time_sec": navigation.mailbox.snapshot.first_own_foot_time_sec,
                "first_own_foot": navigation.mailbox.snapshot.first_own_foot,
                "first_contact_relative_y_mps": (
                    navigation.mailbox.snapshot.first_contact_relative_y_mps
                ),
                "prefoot_nonfoot_count": navigation.mailbox.snapshot.prefoot_nonfoot_count,
                "post_active_frames": navigation.post_active_frames,
                "contract_hash": navigation.contract_hash,
                "mailbox_contract_hash": navigation.mailbox.contract_hash,
                "evidence_contract_hash": contact_evidence.contract_hash,
            }
            if isinstance(navigation, TeamReceiveContactPhaseActor)
            and navigation.mailbox is not None
            and contact_evidence is not None
            else None
        ),
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
    parser.add_argument("--option-ankle-braking", type=float, choices=(8.0, 12.0, 16.0))
    parser.add_argument("--option-joint-guard-margin", type=float, choices=(0.06, 0.08, 0.10))
    parser.add_argument("--retired-ankle-braking", type=float, choices=(8.0, 12.0, 16.0))
    parser.add_argument("--ball-x", type=float, default=1.92)
    parser.add_argument("--ball-y", type=float, default=-0.80)
    parser.add_argument("--seed", type=int, default=207_200)
    parser.add_argument("--stance", type=float, default=-0.17)
    parser.add_argument("--pass-speed", type=float, default=0.80)
    parser.add_argument("--preview-pass", action="store_true")
    parser.add_argument("--duration", type=float, choices=(8.70, 10.0), default=8.70)
    parser.add_argument("--precontact-pass-standoff", type=float, choices=(0.25, 0.35, 0.45))
    parser.add_argument(
        "--motor-agent", choices=("red.finisher", "red.playmaker"), default="red.finisher"
    )
    parser.add_argument(
        "--motor-entry-frame", type=int, choices=(0, 5, 10, 15, 20, 25, 30), default=30
    )
    parser.add_argument("--strike-through", type=float, choices=(0.0, 0.08, 0.16), default=0.0)
    parser.add_argument("--directed-pass-speed", type=float, choices=(1.0, 1.5, 2.0), default=0.0)
    parser.add_argument(
        "--pass-lateral-bias", type=float, choices=(-0.40, -0.20, 0.0, 0.20, 0.40), default=0.0
    )
    parser.add_argument("--dual-receiver-motor", action="store_true")
    parser.add_argument(
        "--handoff-profile",
        choices=("legacy", "strict", "tracking", "committed"),
        default="legacy",
    )
    parser.add_argument(
        "--receive-profile",
        choices=("legacy", "lateral", "retention", "lateral_retention"),
        default="legacy",
    )
    parser.add_argument(
        "--phase-profile",
        choices=("default", "fast", "predictive", "ultra"),
        default="default",
    )
    parser.add_argument(
        "--navigation-profile",
        choices=(
            "none",
            "receive_tap",
            "receive_phase",
            "receive_contact",
            "follow",
            "lead",
            "damped",
            "wide",
            "lease1",
            "lease2",
            "lease3",
            "lease2_damped",
        ),
        default="none",
    )
    parser.add_argument(
        "--teacher-profile", choices=("default", "live_after_receive"), default="default"
    )
    parser.add_argument(
        "--receive-teacher-profile",
        choices=("default", "neutral", "soft", "cushion", "combined"),
        default="default",
    )
    parser.add_argument(
        "--capture-profile", choices=("none", "short", "medium", "long"), default="none"
    )
    parser.add_argument("--capture-b6-microphysics", action="store_true")
    parser.add_argument("--navigation-phase-weights", type=float, nargs=5)
    parser.add_argument("--navigation-post-weights", type=float, nargs=5)
    parser.add_argument(
        "--velocity-cushion-gain", type=float, choices=(0.0, 0.5, 1.0, 2.0), default=0.0
    )
    parser.add_argument(
        "--receive-velocity-prediction-sec",
        type=float,
        choices=(0.0, 0.04, 0.08, 0.12),
        default=0.0,
    )
    parser.add_argument("--receive-velocity-prediction-lateral-only", action="store_true")
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
        option_ankle_braking=args.option_ankle_braking,
        option_joint_guard_margin_rad=args.option_joint_guard_margin,
        retired_ankle_braking=args.retired_ankle_braking,
        ball_x_m=args.ball_x,
        ball_y_m=args.ball_y,
        seed=args.seed,
        stance_lateral_m=args.stance,
        pass_speed_mps=args.pass_speed,
        preview_pass=args.preview_pass,
        duration_sec=args.duration,
        precontact_pass_standoff_m=args.precontact_pass_standoff,
        motor_agent_id=args.motor_agent,
        motor_entry_frame=args.motor_entry_frame,
        strike_through_m=args.strike_through,
        directed_pass_speed_mps=args.directed_pass_speed,
        pass_lateral_bias_m=args.pass_lateral_bias,
        dual_receiver_motor=args.dual_receiver_motor,
        velocity_cushion_gain=args.velocity_cushion_gain,
        receive_velocity_prediction_sec=args.receive_velocity_prediction_sec,
        receive_velocity_prediction_lateral_only=args.receive_velocity_prediction_lateral_only,
        handoff_profile=args.handoff_profile,
        receive_profile=args.receive_profile,
        phase_profile=args.phase_profile,
        navigation_profile=args.navigation_profile,
        navigation_phase_weights=(
            None if args.navigation_phase_weights is None else tuple(args.navigation_phase_weights)
        ),
        navigation_post_weights=(
            None if args.navigation_post_weights is None else tuple(args.navigation_post_weights)
        ),
        teacher_profile=args.teacher_profile,
        receive_teacher_profile=args.receive_teacher_profile,
        capture_profile=args.capture_profile,
        capture_b6_microphysics=args.capture_b6_microphysics,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
