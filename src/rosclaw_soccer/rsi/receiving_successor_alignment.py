"""SIM high-level receiving-to-kick alignment teacher on current state only.

Outputs body velocity commands for the existing neural locomotor, not joint
targets or Cartesian foot overrides. The narrow geometry is a NEW candidate
task condition, not a relabeling of clean-foot reception or old kick exams.
No success/generalization claim follows from this analytic teacher alone.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SuccessorAlignment:
    command_body: tuple[float, float, float]
    ball_body_xy: tuple[float, float]
    relative_speed_xy: float
    goal_yaw_error: float
    geometry_ready: bool


def successor_alignment(
    qpos: np.ndarray, qvel: np.ndarray, return_goal: np.ndarray
) -> SuccessorAlignment:
    for value, shape in ((qpos, (43,)), (qvel, (41,)), (return_goal, (3,))):
        if (
            not isinstance(value, np.ndarray)
            or value.shape != shape
            or not np.isfinite(value).all()
        ):
            raise ValueError("finite current canonical state and declared current goal required")
    if abs(np.linalg.norm(qpos[3:7]) - 1) > 1e-5:
        raise ValueError("unit actual body orientation required")
    qw, x, y, z = qpos[3:7]
    yaw = float(np.arctan2(2 * (qw * z + x * y), 1 - 2 * (y * y + z * z)))
    c, s = np.cos(yaw), np.sin(yaw)
    body_from_world = np.array([[c, s], [-s, c]])
    delta = body_from_world @ (qpos[36:38] - qpos[:2])
    ball_velocity = body_from_world @ qvel[35:37]
    relative = float(np.linalg.norm(qvel[35:37] - qvel[:2]))
    goal = body_from_world @ (return_goal[:2] - qpos[:2])
    if np.linalg.norm(goal) < 0.5:
        raise ValueError("nondegenerate current return task required")
    error = float(np.arctan2(goal[1], goal[0]))
    # Track the measured ball while placing the body behind the proven
    # full-reference entry region, rather than freezing at any <=.65m range.
    velocity = ball_velocity + 1.2 * (delta - np.array([0.12, 0.03]))
    norm = float(np.linalg.norm(velocity))
    if norm > 0.35:
        velocity *= 0.35 / norm
    command = (
        float(np.clip(velocity[0], -0.35, 0.35)),
        float(np.clip(velocity[1], -0.35, 0.35)),
        float(np.clip(1.5 * error, -0.6, 0.6)),
    )
    ready = bool(
        0.06 <= delta[0] <= 0.22
        and -0.04 <= delta[1] <= 0.09
        and relative <= 0.4
        and abs(error) <= 0.2
        and qpos[38] <= 0.3
        and qpos[2] >= 0.65
        and 1 - 2 * (qpos[4] ** 2 + qpos[5] ** 2) >= 0.9
    )
    return SuccessorAlignment(command, (float(delta[0]), float(delta[1])), relative, error, ready)


def assigned_receiver_pass_goal(anchor_xy: np.ndarray, pass_direction: np.ndarray) -> np.ndarray:
    if (
        any(
            not isinstance(a, np.ndarray) or a.shape != (2,) or not np.isfinite(a).all()
            for a in (anchor_xy, pass_direction)
        )
        or abs(np.linalg.norm(pass_direction) - 1) > 1e-6
    ):
        raise ValueError("finite communicated anchor and unit declared pass direction required")
    lateral = np.array([-pass_direction[1], pass_direction[0]])
    point = anchor_xy - 0.25 * pass_direction - 0.12 * lateral
    return np.array([point[0], point[1], 0.11], np.float64)


class SuccessorAlignmentGate:
    """Causal geometry-teacher phase, advanced once per ACTUAL control.

    Disabled means the original receiving predicate exactly, not a narrower
    reinterpretation. Enabled waits for original contact/readiness, moves the
    body using the frozen neural locomotor, then requires10 measured samples.
    """

    def __init__(self, *, enabled: bool) -> None:
        if type(enabled) is not bool:
            raise ValueError("explicit alignment experiment arm required")
        self.enabled = enabled
        self.active = False
        self.consecutive = 0
        self.last_control = 0

    def observe(
        self,
        qpos: np.ndarray,
        qvel: np.ndarray,
        return_goal: np.ndarray,
        *,
        control: int,
        receiving_ready: bool,
        contact_eligible: bool,
        body_healthy: bool,
    ) -> bool:
        if (
            type(control) is not int
            or control != self.last_control + 1
            or any(type(x) is not bool for x in (receiving_ready, contact_eligible, body_healthy))
        ):
            raise ValueError("one consecutive actual measured control required")
        self.last_control = control
        if not self.enabled:
            return receiving_ready
        if receiving_ready:
            self.active = True
        if not self.active:
            return False
        result = successor_alignment(qpos, qvel, return_goal)
        if np.linalg.norm(result.ball_body_xy) > 0.8:
            self.active = False
            self.consecutive = 0
            return False
        good = result.geometry_ready and contact_eligible and body_healthy
        self.consecutive = self.consecutive + 1 if good else 0
        return self.consecutive >= 10
