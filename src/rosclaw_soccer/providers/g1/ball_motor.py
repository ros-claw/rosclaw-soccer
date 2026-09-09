"""Frozen G1 ball-conditioned residual proposals for simulation controllers.

Each player owns a separate instance and episode history. This reads canonical
course observations and frozen teacher targets; it neither steps a simulator
nor grants authority. A caller still owns teacher history, course transforms,
joint/torque guards, physical contact evidence and task admission.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from rosclaw_soccer.providers.g1.approach_router import load_bounded_reference_parameters
from rosclaw_soccer.providers.g1.proprioceptive_router import g1_approach_proprioception
from rosclaw_soccer.sim.contracts import hash_json
from rosclaw_soccer.training.ball_residual import BallResidualEnvelope


def _vector(value: np.ndarray, size: int) -> np.ndarray:
    result = np.asarray(value)
    if result.shape != (size,) or result.dtype.kind not in "fi" or not np.isfinite(result).all():
        raise ValueError("finite numeric motor vector required")
    return result.astype(np.float64)


def g1_ball_motor_observation(
    *,
    course_qpos: np.ndarray,
    course_qvel: np.ndarray,
    default_angles: np.ndarray,
    teacher_target: np.ndarray,
    previous_residual: np.ndarray,
    frame: int,
    episode_frames: int = 200,
) -> np.ndarray:
    """Exact 133-feature training contract, including its declared [-10,10] clip.

    Clipping is model preprocessing, NOT an out-of-domain admission decision.
    This fixed-course model has no requested goal input or arbitrary-yaw claim.
    """
    if (
        type(frame) is not int
        or type(episode_frames) is not int
        or episode_frames != 200
        or not 0 <= frame < episode_frames
    ):
        raise ValueError("motor contract requires frames 0..199 at 50 Hz")
    g1_approach_proprioception(course_qpos, course_qvel)
    q, v = _vector(course_qpos, 43), _vector(course_qvel, 41)
    default = _vector(default_angles, 29)
    teacher, previous = _vector(teacher_target, 29), _vector(previous_residual, 29)
    phase = frame / episode_frames * 6.2831853
    w, x, y, z = q[3:7]
    gravity = [2 * (w * y - x * z), -2 * (w * x + y * z), 2 * (x * x + y * y) - 1]
    observation: np.ndarray = np.concatenate(
        (
            q[7:36] - default,
            v[6:35] * 0.1,
            gravity,
            v[:3],
            v[3:6] * 0.2,
            q[36:39] - q[:3],
            v[35:38],
            [np.sin(phase), np.cos(phase)],
            previous,
            teacher - default,
        )
    )
    if not np.isfinite(observation).all():
        raise ValueError("motor feature conversion overflow")
    return observation.clip(-10, 10).astype(np.float32)


@dataclass(frozen=True)
class BallMotorProposal:
    frame: int
    target_rad: tuple[float, ...]
    residual_rad: tuple[float, ...]
    observation_hash: str
    contract_hash: str
    activation_ceiling: str = "SIM_ONLY"


class G1FrozenBallMotor:
    """Numeric-only frozen actor with per-player, consecutive-tick state.

    A fault latches until an explicit new episode. Duplicate/skipped ticks must
    never advance muscle history invisibly. No optimizer or Torch import here.
    """

    def __init__(
        self,
        weights: Path,
        *,
        expected_actor_hash: str,
        foundation_hash: str,
        reference_library_hash: str,
        default_angles: np.ndarray,
        envelope: BallResidualEnvelope | None = None,
    ) -> None:
        for value in (expected_actor_hash, foundation_hash, reference_library_hash):
            if not isinstance(value, str) or re.fullmatch(r"sha256:[0-9a-f]{64}", value) is None:
                raise ValueError("explicit motor, foundation and reference hashes required")
        self.envelope = envelope or BallResidualEnvelope()
        if self.envelope != BallResidualEnvelope():
            raise ValueError("frozen motor requires its original trained residual envelope")
        shapes: dict[str, tuple[int, ...]] = {"logstd": (29,)}
        for head, output in (("actor", 29), ("critic", 1)):
            for layer, n_in, n_out in ((0, 133, 128), (2, 128, 128), (4, 128, output)):
                shapes[f"{head}.{layer}.weight"] = (n_out, n_in)
                shapes[f"{head}.{layer}.bias"] = (n_out,)
        self._parameters, self.policy_hash = load_bounded_reference_parameters(
            weights, expected_actor_hash, shapes
        )
        self._default = _vector(default_angles, 29).copy()
        self._default.setflags(write=False)
        self.contract_hash = str(
            hash_json(
                {
                    "schema": "rosclaw_soccer.g1_frozen_ball_motor.v1",
                    "actor_hash": self.policy_hash,
                    "foundation_hash": foundation_hash,
                    "reference_library_hash": reference_library_hash,
                    "default_angles": self._default.tolist(),
                    "envelope": asdict(self.envelope),
                    "observation": "g1_course_ball_residual_133.v1",
                    "control_dt_sec": 0.02,
                    "episode_frames": 200,
                    "activation_ceiling": "SIM_ONLY",
                }
            )
        )
        self._previous = np.zeros(29, dtype=np.float32)
        self._frame = 0
        self._active = False
        self._faulted = False

    def begin_episode(self) -> None:
        """Reset only this player's proposal history, never world or teacher state."""
        self._previous = np.zeros(29, dtype=np.float32)
        self._frame = 0
        self._active = True
        self._faulted = False

    def propose(
        self,
        *,
        frame: int,
        course_qpos: np.ndarray,
        course_qvel: np.ndarray,
        teacher_target: np.ndarray,
    ) -> BallMotorProposal:
        if not self._active or self._faulted:
            raise ValueError("start a declared motor episode before proposing")
        try:
            if type(frame) is not int or frame != self._frame:
                raise ValueError("consecutive motor control frames required")
            observation = g1_ball_motor_observation(
                course_qpos=course_qpos,
                course_qvel=course_qvel,
                default_angles=self._default,
                teacher_target=teacher_target,
                previous_residual=self._previous,
                frame=frame,
            )
            p = self._parameters
            x = observation
            for layer in (0, 2):
                x = np.tanh(p[f"actor.{layer}.weight"] @ x + p[f"actor.{layer}.bias"])
            raw = p["actor.4.weight"] @ x + p["actor.4.bias"]
            desired = self.envelope.maximum_offset_rad * np.tanh(raw)
            residual = self._previous + np.clip(
                self.envelope.smoothing * (desired - self._previous),
                -self.envelope.maximum_step_rad,
                self.envelope.maximum_step_rad,
            )
            target = _vector(teacher_target, 29) + residual
            if not np.isfinite(raw).all() or not np.isfinite(target).all():
                raise ValueError("nonfinite frozen motor proposal")
            proposal = BallMotorProposal(
                frame=frame,
                target_rad=tuple(float(v) for v in target),
                residual_rad=tuple(float(v) for v in residual),
                observation_hash=str(hash_json(observation.tolist())),
                contract_hash=self.contract_hash,
            )
        except (ValueError, TypeError, OverflowError, FloatingPointError):
            self._faulted = True
            raise
        self._previous = residual.copy()
        self._frame += 1
        if self._frame == 200:
            self._active = False
        return proposal

