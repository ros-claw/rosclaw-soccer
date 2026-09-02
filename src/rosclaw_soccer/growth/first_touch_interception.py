"""Proprioceptive ball-foot interception reflex for moving-ball First Touch."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from rosclaw_soccer.sim.contracts import hash_json


@dataclass(frozen=True)
class FirstTouchInterceptionConfig:
    """Bounded task-space feedback decoded through the measured foot Jacobian."""

    strike_ankle_offset_m: tuple[float, float, float] = (-0.10, 0.21, -0.07)
    prediction_horizon_sec: float = 0.04
    position_gain_n_per_m: float = 60.0
    velocity_damping_n_per_mps: float = 3.0
    maximum_task_force_n: float = 24.0
    maximum_joint_residual_nm: float = 6.0
    maximum_foot_ball_distance_m: float = 0.60
    start_policy_frame: int = 232
    end_policy_frame: int = 252
    activation_ceiling: str = "SIM_ONLY"
    hardware_authorized: bool = False
    schema_version: str = "rosclaw_soccer.first_touch_interception_config.v1"

    def __post_init__(self) -> None:
        values = (
            *self.strike_ankle_offset_m,
            self.prediction_horizon_sec,
            self.position_gain_n_per_m,
            self.velocity_damping_n_per_mps,
            self.maximum_task_force_n,
            self.maximum_joint_residual_nm,
            self.maximum_foot_ball_distance_m,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("First Touch interception settings must be finite")
        if float(np.linalg.norm(self.strike_ankle_offset_m)) > 0.30:
            raise ValueError("First Touch interception strike offset is invalid")
        if not 0.0 <= self.prediction_horizon_sec <= 0.10:
            raise ValueError("First Touch interception prediction horizon is invalid")
        if not 5.0 <= self.position_gain_n_per_m <= 200.0:
            raise ValueError("First Touch interception position gain is invalid")
        if not 0.0 <= self.velocity_damping_n_per_mps <= 20.0:
            raise ValueError("First Touch interception velocity damping is invalid")
        if not 2.0 <= self.maximum_task_force_n <= 60.0:
            raise ValueError("First Touch interception force limit is invalid")
        if not 1.0 <= self.maximum_joint_residual_nm <= 10.0:
            raise ValueError("First Touch interception torque limit is invalid")
        if not 0.25 <= self.maximum_foot_ball_distance_m <= 0.80:
            raise ValueError("First Touch interception proximity is invalid")
        if not 210 <= self.start_policy_frame < self.end_policy_frame <= 275:
            raise ValueError("First Touch interception policy window is invalid")
        if self.activation_ceiling != "SIM_ONLY" or self.hardware_authorized:
            raise ValueError("First Touch interception reflex must remain SIM_ONLY")

    @property
    def config_hash(self) -> str:
        return str(hash_json(asdict(self)))


@dataclass(frozen=True)
class FirstTouchInterceptionEffect:
    torque: NDArray[np.float64]
    task_force_n: NDArray[np.float64]
    position_error_m: NDArray[np.float64]
    foot_ball_distance_m: float
    active: bool


def first_touch_interception_effect(
    *,
    model: Any,
    data: Any,
    striking_ankle_body_id: int,
    actuated_dof_indices: NDArray[np.int64],
    ball_position_m: NDArray[np.float64],
    ball_velocity_mps: NDArray[np.float64],
    policy_frame: int,
    contact_observed: bool,
    kick_foot: str,
    config: FirstTouchInterceptionConfig,
) -> FirstTouchInterceptionEffect:
    """Evaluate one causal reflex without bypassing the world safety projection."""

    indices = np.asarray(actuated_dof_indices, dtype=np.int64)
    ball = np.asarray(ball_position_m, dtype=np.float64)
    velocity = np.asarray(ball_velocity_mps, dtype=np.float64)
    if (
        indices.shape != (29,)
        or len(set(int(value) for value in indices)) != 29
        or np.any(indices < 0)
        or np.any(indices >= int(model.nv))
    ):
        raise ValueError("First Touch interception requires 29 unique actuated DoFs")
    if (
        ball.shape != (3,)
        or velocity.shape != (3,)
        or not all(np.all(np.isfinite(value)) for value in (ball, velocity))
    ):
        raise ValueError("First Touch interception ball state is invalid")
    if kick_foot not in {"left", "right"}:
        raise ValueError("First Touch interception kick foot is invalid")
    zero = np.zeros(29, dtype=np.float64)
    zero_xyz = np.zeros(3, dtype=np.float64)
    foot = np.asarray(data.xpos[striking_ankle_body_id], dtype=np.float64)
    distance = float(np.linalg.norm(foot - ball))
    if (
        contact_observed
        or not config.start_policy_frame <= policy_frame <= config.end_policy_frame
        or distance > config.maximum_foot_ball_distance_m
    ):
        return FirstTouchInterceptionEffect(zero, zero_xyz, zero_xyz, distance, False)

    import mujoco

    jacobian = np.zeros((3, int(model.nv)), dtype=np.float64)
    rotation = np.zeros((3, int(model.nv)), dtype=np.float64)
    mujoco.mj_jac(
        model,
        data,
        jacobian,
        rotation,
        foot,
        striking_ankle_body_id,
    )
    foot_velocity = jacobian @ np.asarray(data.qvel, dtype=np.float64)
    offset = np.asarray(config.strike_ankle_offset_m, dtype=np.float64).copy()
    if kick_foot == "left":
        offset[1] *= -1.0
    desired = ball + config.prediction_horizon_sec * velocity + offset
    error = desired - foot
    force = config.position_gain_n_per_m * error - config.velocity_damping_n_per_mps * foot_velocity
    norm = float(np.linalg.norm(force))
    if norm > config.maximum_task_force_n:
        force *= config.maximum_task_force_n / norm
    torque = jacobian[:, indices].T @ force
    torque = np.clip(
        torque,
        -config.maximum_joint_residual_nm,
        config.maximum_joint_residual_nm,
    )
    return FirstTouchInterceptionEffect(torque, force, error, distance, True)


__all__ = [
    "FirstTouchInterceptionConfig",
    "FirstTouchInterceptionEffect",
    "first_touch_interception_effect",
]
