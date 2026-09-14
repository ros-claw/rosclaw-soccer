"""Opt-in SIM-only demonstration tracking with causal handoff to live feedback.

This fixed-clock adapter is not a neural policy, an executor, or a stability
certificate. Callers must validate body/root/scene support and retain downstream
residual, joint and actuator guards. Demonstration PD proposals are NOT delivered
torques. Each role/episode needs a separate tracker.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from numbers import Real

import numpy as np


def _real(value: object) -> bool:
    try:
        return (
            isinstance(value, Real)
            and not isinstance(value, (bool, np.bool_))
            and math.isfinite(value)
        )
    except OverflowError:
        return False


def _array(value: np.ndarray, shape: tuple[int, ...] | None = None) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype.kind not in "fiu" or not np.isfinite(array).all():
        raise ValueError("memory requires finite real arrays")
    if shape is not None and array.shape != shape:
        raise ValueError("memory array shape mismatch")
    result = np.array(array, dtype=np.float64, copy=True, order="C")
    result.flags.writeable = False
    return result


@dataclass(frozen=True)
class JointControlMemory:
    """Prior demonstration in explicit caller-defined joint order, starting at t=0."""

    joint_position: np.ndarray
    joint_velocity: np.ndarray
    pd_proposal: np.ndarray
    kp: np.ndarray
    kd: np.ndarray
    period_sec: float
    joint_names: tuple[str, ...]
    activation_ceiling: str = "SIM_ONLY"

    def __post_init__(self) -> None:
        if not _real(self.period_sec) or self.period_sec <= 0:
            raise ValueError("memory period must be positive and finite")
        names = tuple(self.joint_names)
        if (
            not names
            or any(not isinstance(n, str) or not n.strip() for n in names)
            or len(set(names)) != len(names)
        ):
            raise ValueError("memory joint names must be explicit, unique and nonempty")
        q = _array(self.joint_position)
        if q.ndim != 2 or q.shape[0] < 1 or q.shape[1] != len(names):
            raise ValueError("memory requires a nonempty samples-by-joints matrix")
        for name in ("joint_position", "joint_velocity", "pd_proposal", "kp", "kd"):
            value = _array(getattr(self, name), q.shape)
            if name in ("kp", "kd") and np.any(value < 0):
                raise ValueError("memory gains must be nonnegative")
            object.__setattr__(self, name, value)
        if self.activation_ceiling != "SIM_ONLY":
            raise ValueError("control memory is SIM_ONLY")
        object.__setattr__(self, "joint_names", names)
        object.__setattr__(self, "period_sec", float(self.period_sec))

    @property
    def content_hash(self) -> str:
        import json

        digest = hashlib.sha256(
            json.dumps(
                {
                    "joint_names": self.joint_names,
                    "period_sec": self.period_sec,
                    "shape": self.joint_position.shape,
                    "activation_ceiling": self.activation_ceiling,
                },
                sort_keys=True,
            ).encode()
        )
        for value in (self.joint_position, self.joint_velocity, self.pd_proposal, self.kp, self.kd):
            digest.update(value.astype("<f8", copy=False).tobytes())
        return "sha256:" + digest.hexdigest()


@dataclass(frozen=True)
class JointMemoryTrackingConfig:
    feedback_scale: float
    correction_cap_nm: float
    release_delay_sec: float
    release_fade_sec: float
    initial_position_tolerance_rad: float
    initial_velocity_tolerance_rad_s: float

    def __post_init__(self) -> None:
        values = (
            self.feedback_scale,
            self.correction_cap_nm,
            self.release_delay_sec,
            self.release_fade_sec,
            self.initial_position_tolerance_rad,
            self.initial_velocity_tolerance_rad_s,
        )
        if (
            not all(_real(v) and v >= 0 for v in values)
            or self.correction_cap_nm <= 0
            or self.release_fade_sec <= 0
        ):
            raise ValueError(
                "tracking config requires finite explicit bounds and positive cap/fade"
            )


@dataclass(frozen=True)
class JointMemoryProposal:
    pd_proposal: np.ndarray
    correction_nm: np.ndarray
    memory_weight: float
    observed_contact_timestamp_sec: float | None
    sample_index: int


class JointMemoryTracker:
    """Sequential fixed-clock proposals. Invalid input faults the episode closed."""

    def __init__(self, memory: JointControlMemory, config: JointMemoryTrackingConfig) -> None:
        self.memory = memory
        self.config = config
        self._index = 0
        self._contact_sec: float | None = None
        self._faulted = False

    def sample(
        self,
        *,
        timestamp_sec: float,
        joint_position: np.ndarray,
        joint_velocity: np.ndarray,
        live_pd_proposal: np.ndarray,
        observed_contact: bool,
    ) -> JointMemoryProposal:
        if self._faulted:
            raise ValueError("memory tracker is fault-latched; start a new episode")
        try:
            return self._sample(
                timestamp_sec, joint_position, joint_velocity, live_pd_proposal, observed_contact
            )
        except (ValueError, TypeError, OverflowError):
            self._faulted = True
            raise

    def _sample(
        self,
        timestamp: float,
        joint_position: np.ndarray,
        joint_velocity: np.ndarray,
        live_pd_proposal: np.ndarray,
        contact: bool,
    ) -> JointMemoryProposal:
        i = self._index
        if (
            not _real(timestamp)
            or timestamp < 0
            or type(contact) is not bool
            or i >= len(self.memory.joint_position)
            or abs(timestamp - i * self.memory.period_sec) > 1e-9
        ):
            raise ValueError(
                "memory requires sequential supported native timestamps and a boolean event"
            )
        shape = (len(self.memory.joint_names),)
        q, dq, live = (_array(v, shape) for v in (joint_position, joint_velocity, live_pd_proposal))
        if i == 0 and (
            np.max(np.abs(q - self.memory.joint_position[0]))
            > self.config.initial_position_tolerance_rad
            or np.max(np.abs(dq - self.memory.joint_velocity[0]))
            > self.config.initial_velocity_tolerance_rad_s
        ):
            raise ValueError("initial joint state is outside demonstration support")
        if contact and self._contact_sec is None:
            self._contact_sec = float(timestamp)
        elapsed = 0.0 if self._contact_sec is None else float(timestamp) - self._contact_sec
        weight = float(
            np.clip(
                1.0 - (elapsed - self.config.release_delay_sec) / self.config.release_fade_sec,
                0.0,
                1.0,
            )
        )
        with np.errstate(over="raise", invalid="raise"):
            try:
                correction = np.clip(
                    self.config.feedback_scale
                    * (
                        self.memory.kp[i] * (self.memory.joint_position[i] - q)
                        + self.memory.kd[i] * (self.memory.joint_velocity[i] - dq)
                    ),
                    -self.config.correction_cap_nm,
                    self.config.correction_cap_nm,
                )
                proposal = (
                    weight * (self.memory.pd_proposal[i] + correction) + (1.0 - weight) * live
                )
            except FloatingPointError as exc:
                raise ValueError("nonfinite memory arithmetic") from exc
        self._index += 1
        proposal.flags.writeable = False
        correction.flags.writeable = False
        return JointMemoryProposal(proposal, correction, weight, self._contact_sec, i)
