"""Exclusive SIM_ONLY whole-leg receiving motor over immutable current foot state."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from rosclaw_soccer.rsi.team_receive_contact_evidence import ReceiveContactMailbox
from rosclaw_soccer.sim.contracts import hash_json
from rosclaw_soccer.skills.team.motor_option import TeamMotorObservation, TeamMotorTarget


@dataclass
class ReceivingTaskspaceMotor:
    """Unique motor owner; proposes guarded joint targets, never raw torque.

    A zero-gain instance is the ownership-matched parent. The shared world
    retains joint, torque, collision, and failure guards. No hardware path.
    """

    agent_id: str
    mailbox: ReceiveContactMailbox
    pre_gain: float
    post_gain: float
    velocity_horizon_sec: float
    hip_clearance_rad: float = 0.0
    ankle_compensation_rad: float = 0.0
    idle_before_first_touch: bool = False
    activation_ceiling: str = field(init=False, default="SIM_ONLY")
    needs_foot_kinematics: bool = field(init=False, default=True)
    contract_hash: str = field(init=False)
    _next_frame: int = field(init=False, default=0)
    _nonzero_target_frames: int = field(init=False, default=0)
    _peak_correction_rad: float = field(init=False, default=0.0)

    def __post_init__(self) -> None:
        values = (
            self.pre_gain,
            self.post_gain,
            self.velocity_horizon_sec,
            self.hip_clearance_rad,
            self.ankle_compensation_rad,
        )
        if (
            not isinstance(self.mailbox, ReceiveContactMailbox)
            or self.mailbox.agent_id != self.agent_id
            or any(type(value) is not float or not math.isfinite(value) for value in values)
            or not 0 <= self.pre_gain <= 1
            or not 0 <= self.post_gain <= 1
            or not 0 <= self.velocity_horizon_sec <= 0.15
            or not -0.2 <= self.hip_clearance_rad <= 0.2
            or not -0.2 <= self.ankle_compensation_rad <= 0.2
            or type(self.idle_before_first_touch) is not bool
        ):
            raise ValueError("bounded same-player SIM_ONLY receiving motor required")
        self.contract_hash = hash_json(
            {
                "schema": "rosclaw_soccer.rsi.receiving_taskspace_motor.v1",
                "agent_id": self.agent_id,
                "mailbox_hash": self.mailbox.contract_hash,
                "pre_gain": self.pre_gain,
                "post_gain": self.post_gain,
                "velocity_horizon_sec": self.velocity_horizon_sec,
                "hip_clearance_rad": self.hip_clearance_rad,
                "ankle_compensation_rad": self.ankle_compensation_rad,
                "idle_before_first_touch": self.idle_before_first_touch,
                "target_correction_limit_rad": 0.25,
                "activation_ceiling": "SIM_ONLY",
            }
        )

    @property
    def nonzero_target_frames(self) -> int:
        return self._nonzero_target_frames

    @property
    def peak_correction_rad(self) -> float:
        return self._peak_correction_rad

    def propose(self, observation: TeamMotorObservation) -> TeamMotorTarget | None:
        if not isinstance(observation, TeamMotorObservation):
            raise ValueError("typed motor observation required")
        observation.__post_init__()
        feet = observation.foot_kinematics
        foundation = observation.foundation
        snapshot = self.mailbox.snapshot
        if (
            observation.agent_id != self.agent_id
            or observation.frame != self._next_frame
            or foundation is None
            or feet is None
            or feet.foot_linear_velocity_world_mps is None
            or abs(snapshot.physics_time_sec - observation.time_sec) > 0.002001
        ):
            raise ValueError("consecutive same-player current foot/foundation/contact required")
        self._next_frame += 1
        base = foundation.target
        first_time = snapshot.first_own_foot_time_sec
        if first_time is None:
            if self.idle_before_first_touch:
                return None
            right = observation.qpos[37] < observation.qpos[1]
            gain = self.pre_gain
            active = True
        else:
            if snapshot.first_own_foot not in ("left_foot", "right_foot"):
                raise ValueError("measured receiving foot required")
            right = snapshot.first_own_foot == "right_foot"
            gain = self.post_gain
            active = observation.time_sec - first_time <= 1.2
        if not active or gain == 0 and self.hip_clearance_rad == self.ankle_compensation_rad == 0:
            return None if self.idle_before_first_touch else base
        side = 1 if right else 0
        foot = np.asarray(feet.foot_position_world_m[side], dtype=np.float64)
        foot_velocity = np.asarray(feet.foot_linear_velocity_world_mps[side], dtype=np.float64)
        ball = np.asarray(observation.qpos[36:39], dtype=np.float64)
        ball_velocity = np.asarray(observation.qvel[35:38], dtype=np.float64)
        if np.linalg.norm(ball - foot) > 0.85 or observation.qpos[2] < 0.55:
            return None if self.idle_before_first_touch else base
        target_foot = ball + np.asarray((-0.18, -0.03 if right else 0.03, -0.08))
        target_foot[2] = np.clip(target_foot[2], 0.04, 0.08)
        error = np.clip(target_foot - foot, -0.2, 0.2)
        velocity_error = np.clip(ball_velocity - foot_velocity, -1.5, 1.5)
        task_delta = error + self.velocity_horizon_sec * velocity_error
        jacobian = np.asarray(feet.foot_linear_jacobian_world[side], dtype=np.float64)
        step = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + 0.02 * np.eye(3), task_delta)
        correction = np.clip(gain * step, -0.25, 0.25)
        correction[0] += self.hip_clearance_rad
        correction[4] += self.ankle_compensation_rad
        np.clip(correction, -0.25, 0.25, out=correction)
        offset = 6 if right else 0
        limits = np.asarray(feet.leg_joint_limits_rad[side], dtype=np.float64)
        target = np.asarray(base.target_rad, dtype=np.float64)
        target[offset : offset + 6] = np.clip(
            target[offset : offset + 6] + correction,
            limits[:, 0],
            limits[:, 1],
        )
        actual_correction = (
            target[offset : offset + 6] - np.asarray(base.target_rad)[offset : offset + 6]
        )
        peak = float(np.max(np.abs(actual_correction)))
        if peak > 0:
            self._nonzero_target_frames += 1
            self._peak_correction_rad = max(self._peak_correction_rad, peak)
        return TeamMotorTarget(tuple(float(x) for x in target), base.kp, base.kd)
