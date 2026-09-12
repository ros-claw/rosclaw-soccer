"""Frozen target-conditioned proposals with private per-player episode state.

No Torch, optimizer, simulator stepping, transport or activation authority.
The caller owns frame transforms, teacher history, admission and physical guards.
"""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from rosclaw_soccer.providers.g1.approach_router import load_bounded_reference_parameters
from rosclaw_soccer.providers.g1.ball_motor import _vector, g1_ball_motor_observation
from rosclaw_soccer.sim.contracts import hash_json
from rosclaw_soccer.training.ball_residual import BallResidualEnvelope
from rosclaw_soccer.training.goal_reference_actor import ReferenceHeadingEnvelope


@dataclass(frozen=True)
class GoalReferenceMotorProposal:
    frame: int
    target_rad: tuple[float, ...]
    residual_rad: tuple[float, ...]
    reference_heading_used_rad: float
    next_reference_heading_rad: float
    observation_hash: str
    contract_hash: str
    episode_hash: str
    activation_ceiling: str = "SIM_ONLY"


class G1FrozenGoalReferenceMotor:
    """136-feature/30-output numeric checkpoint, not a direct-torque policy.

    An episode fixes its target in the same declared frame as observations.
    Proposal n consumes teacher targets generated with heading n, and returns
    heading n+1 for the NEXT teacher tick. No automatic teacher mutation occurs.
    Numerical and physical parity must be verified before integrating a model;
    content hashes alone do not qualify its behavior or authorize deployment.
    """

    def __init__(
        self,
        weights: Path,
        *,
        expected_actor_hash: str,
        foundation_hash: str,
        reference_library_hash: str,
        course_transform_hash: str,
        default_angles: NDArray[Any],
        reference_center_rad: float = 0.0,
    ) -> None:
        if (
            type(reference_center_rad) not in (int, float)
            or not math.isfinite(reference_center_rad)
            or abs(reference_center_rad) > 0.6
        ):
            raise ValueError("finite declared reference center within +/-0.6 rad required")
        for value in (
            expected_actor_hash,
            foundation_hash,
            reference_library_hash,
            course_transform_hash,
        ):
            if not isinstance(value, str) or re.fullmatch(r"sha256:[0-9a-f]{64}", value) is None:
                raise ValueError("explicit motor, foundation, reference and frame hashes required")
        shapes: dict[str, tuple[int, ...]] = {"logstd": (30,)}
        for head, output in (("actor", 29), ("critic", 1)):
            for layer, n_in, n_out in ((0, 133, 128), (2, 128, 128), (4, 128, output)):
                shapes[f"{head}.{layer}.weight"] = (n_out, n_in)
                shapes[f"{head}.{layer}.bias"] = (n_out,)
        for head, n_in, n_out in (
            ("goal_actor", 3, 29),
            ("goal_critic", 3, 1),
            ("reference_actor", 136, 1),
        ):
            shapes[f"{head}.0.weight"] = (32, n_in)
            shapes[f"{head}.0.bias"] = (32,)
            shapes[f"{head}.2.weight"] = (n_out, 32)
            shapes[f"{head}.2.bias"] = (n_out,)
        self._parameters, self.policy_hash = load_bounded_reference_parameters(
            weights, expected_actor_hash, shapes
        )
        self._default = _vector(default_angles, 29).copy()
        self._default.setflags(write=False)
        self._envelope = BallResidualEnvelope()
        self._heading_envelope = ReferenceHeadingEnvelope(
            center_rad=float(reference_center_rad), maximum_offset_rad=0.1
        )
        self.contract_hash = str(
            hash_json(
                {
                    "schema": "rosclaw_soccer.g1_frozen_goal_reference_motor.v2",
                    "actor_hash": self.policy_hash,
                    "foundation_hash": foundation_hash,
                    "reference_library_hash": reference_library_hash,
                    "course_transform_hash": course_transform_hash,
                    "default_angles": self._default.tolist(),
                    "residual_envelope": asdict(self._envelope),
                    "reference_envelope": asdict(self._heading_envelope),
                    "reference_initial_rad": self._heading_envelope.center_rad,
                    "observation": "course_ball_133+target_ray_xy+previous_heading",
                    "control_dt_sec": 0.02,
                    "episode_frames": 200,
                    "activation_ceiling": "SIM_ONLY",
                }
            )
        )
        self._active = False
        self._faulted = False
        self._frame = 0
        self._previous: NDArray[np.float32] = np.zeros(29, dtype=np.float32)
        self._heading: NDArray[np.float32] = np.zeros(1, dtype=np.float32)
        self._target = np.zeros(2, dtype=np.float64)
        self._episode_hash = ""

    def begin_episode(self, *, target_position_xy: NDArray[Any]) -> None:
        """Reset this instance only; invalid resets leave it unavailable."""
        self._active = False
        self._faulted = True
        target = _vector(target_position_xy, 2)
        if np.any(np.abs(target) > 1000):
            raise ValueError("bounded fixed course-frame target required")
        self._target = target.copy()
        self._episode_hash = str(
            hash_json({"contract_hash": self.contract_hash, "target_position_xy": target.tolist()})
        )
        self._previous = np.zeros(29, dtype=np.float32)
        self._heading = np.full(1, self._heading_envelope.center_rad, dtype=np.float32)
        self._frame = 0
        self._faulted = False
        self._active = True

    def _network(
        self, name: str, observation: NDArray[Any], *, body: bool = False
    ) -> NDArray[np.float32]:
        p = self._parameters
        x = np.tanh(p[f"{name}.0.weight"] @ observation + p[f"{name}.0.bias"])
        x = p[f"{name}.2.weight"] @ x + p[f"{name}.2.bias"]
        if body:
            x = p[f"{name}.4.weight"] @ np.tanh(x) + p[f"{name}.4.bias"]
        return cast(NDArray[np.float32], x)

    def propose(
        self,
        *,
        frame: int,
        course_qpos: NDArray[Any],
        course_qvel: NDArray[Any],
        teacher_target: NDArray[Any],
    ) -> GoalReferenceMotorProposal:
        if not self._active or self._faulted:
            raise ValueError("begin a declared goal-reference episode before proposing")
        try:
            if type(frame) is not int or frame != self._frame:
                raise ValueError("consecutive motor control frames required")
            base = g1_ball_motor_observation(
                course_qpos=course_qpos,
                course_qvel=course_qvel,
                default_angles=self._default,
                teacher_target=teacher_target,
                previous_residual=self._previous,
                frame=frame,
            )
            ray = self._target - _vector(course_qpos, 43)[36:38]
            ray /= max(float(np.linalg.norm(ray)), 0.01)
            observation = np.concatenate((base, ray, self._heading)).astype(np.float32)
            with np.errstate(over="raise", invalid="raise"):
                raw = self._network("actor", base, body=True) + self._network(
                    "goal_actor", observation[133:]
                )
                reference_raw = self._network("reference_actor", observation)
                desired = self._envelope.maximum_offset_rad * np.tanh(raw)
                residual = self._previous + np.clip(
                    self._envelope.smoothing * (desired - self._previous),
                    -self._envelope.maximum_step_rad,
                    self._envelope.maximum_step_rad,
                )
                desired_heading = (
                    self._heading_envelope.center_rad
                    + self._heading_envelope.maximum_offset_rad * np.tanh(reference_raw)
                )
                next_heading = self._heading + np.clip(
                    self._heading_envelope.smoothing * (desired_heading - self._heading),
                    -self._heading_envelope.maximum_step_rad,
                    self._heading_envelope.maximum_step_rad,
                )
                target = _vector(teacher_target, 29) + residual
            if not all(np.isfinite(x).all() for x in (raw, reference_raw, target, next_heading)):
                raise ValueError("nonfinite goal-reference proposal")
            proposal = GoalReferenceMotorProposal(
                frame=frame,
                target_rad=tuple(float(x) for x in target),
                residual_rad=tuple(float(x) for x in residual),
                reference_heading_used_rad=float(self._heading[0]),
                next_reference_heading_rad=float(next_heading[0]),
                observation_hash=str(hash_json(observation.tolist())),
                contract_hash=self.contract_hash,
                episode_hash=self._episode_hash,
            )
        except (ValueError, TypeError, OverflowError, FloatingPointError):
            self._faulted = True
            raise
        self._previous = residual.copy()
        self._heading = next_heading.copy()
        self._frame += 1
        self._active = self._frame < 200
        return proposal
