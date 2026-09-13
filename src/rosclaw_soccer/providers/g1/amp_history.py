"""Causal per-player AMP observation history and target decoding, proposals only.

The documented G1 AMP interface is 4 oldest-first frames of 96 observations
and 29 policy-order action values. This module owns no model session, physics,
motor, robot transport, permission or policy promotion. History reconstruction
is not evidence that switching from a different controller is physically safe.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np

from rosclaw_soccer.sim.contracts import hash_json


@dataclass(frozen=True)
class AmpFrameSpec:
    joint_indices: tuple[int, ...]
    default_joint_positions: tuple[float, ...]
    command_low: tuple[float, float, float]
    command_high: tuple[float, float, float]
    angular_velocity_scale: float = 1.0
    joint_position_scale: float = 1.0
    joint_velocity_scale: float = 1.0
    action_scale: float = 0.25
    command_smoothing: float = 0.6

    def __post_init__(self) -> None:
        if (
            type(self.joint_indices) is not tuple
            or len(self.joint_indices) != 29
            or any(type(i) is not int for i in self.joint_indices)
            or sorted(self.joint_indices) != list(range(29))
        ):
            raise ValueError("explicit bijective 29-joint policy mapping required")
        for value, size, bound in (
            (self.default_joint_positions, 29, 10),
            (self.command_low, 3, 10),
            (self.command_high, 3, 10),
        ):
            if (
                type(value) is not tuple
                or len(value) != size
                or any(
                    type(x) not in (int, float) or not math.isfinite(x) or abs(x) > bound
                    for x in value
                )
            ):
                raise ValueError("finite bounded explicit AMP frame configuration required")
        if any(
            not lo <= 0 <= hi or lo == hi
            for lo, hi in zip(self.command_low, self.command_high, strict=True)
        ):
            raise ValueError("command ranges must contain zero and have positive width")
        for scale in (
            self.angular_velocity_scale,
            self.joint_position_scale,
            self.joint_velocity_scale,
            self.action_scale,
        ):
            if type(scale) not in (int, float) or not math.isfinite(scale) or not 0 < scale <= 10:
                raise ValueError("positive bounded observation/action scales required")
        if (
            type(self.command_smoothing) not in (int, float)
            or not math.isfinite(self.command_smoothing)
            or not 0 <= self.command_smoothing < 1
        ):
            raise ValueError("command smoothing must be in [0, 1)")

    @property
    def spec_hash(self) -> str:
        return str(hash_json({"schema": "soccer.amp_frame.v1", **asdict(self)}))


def _vector(value: np.ndarray, size: int, bound: float) -> np.ndarray:
    if (
        not isinstance(value, np.ndarray)
        or value.shape != (size,)
        or value.dtype.kind not in "fiu"
        or not np.isfinite(value).all()
        or np.any(np.abs(value.astype(np.float64)) > bound)
    ):
        raise ValueError("finite bounded numeric AMP vector required")
    return value.astype(np.float64, copy=True)


def _readonly(value: np.ndarray) -> np.ndarray:
    return np.frombuffer(value.astype(np.float32).tobytes(), dtype=np.float32).reshape(value.shape)


@dataclass(frozen=True)
class AmpHistorySnapshot:
    spec_hash: str
    next_tick: int
    history: np.ndarray
    previous_action: np.ndarray
    smoothed_command: np.ndarray


class AmpObservationHistory:
    """One player's 50 Hz history; each prepare must finish with one action.

    Begin only at an explicit fresh episode entry. It repeats the initial
    observation with zero prior action, exactly as the documented AMP entry;
    it must not be used to fabricate a mid-motion history. Restoring a complete
    snapshot requires a new instance. A malformed call latches invalidity.
    Model/body identity and recorded snapshot provenance remain caller-owned.
    """

    def __init__(self, spec: AmpFrameSpec):
        if not isinstance(spec, AmpFrameSpec):
            raise ValueError("explicit AMP frame specification required")
        self._spec = spec
        self._low = np.asarray(spec.command_low, dtype=np.float32)
        self._high = np.asarray(spec.command_high, dtype=np.float32)
        self._indices = np.asarray(spec.joint_indices, dtype=np.int64)
        self._defaults = np.asarray(spec.default_joint_positions, dtype=np.float32)
        self._history = np.zeros((4, 96), dtype=np.float32)
        self._previous = np.zeros(29, dtype=np.float32)
        self._command = np.zeros(3, dtype=np.float32)
        self._next_tick = 0
        self._started = False
        self._pending = False
        self._invalid = False

    @property
    def spec(self) -> AmpFrameSpec:
        return self._spec

    def _check_valid(self) -> None:
        if self._invalid:
            raise RuntimeError("AMP history is invalid")

    def _frame(
        self,
        joint_position: np.ndarray,
        joint_velocity: np.ndarray,
        projected_gravity: np.ndarray,
        angular_velocity: np.ndarray,
    ) -> np.ndarray:
        p = _vector(joint_position, 29, 100)
        v = _vector(joint_velocity, 29, 1000)
        gravity = _vector(projected_gravity, 3, 1.001)
        angular = _vector(angular_velocity, 3, 1000)
        if abs(float(np.linalg.norm(gravity)) - 1) > 1e-4:
            raise ValueError("unit projected gravity required")
        frame: np.ndarray = np.concatenate(
            (
                angular * self.spec.angular_velocity_scale,
                gravity,
                self._command,
                (p[self._indices] - self._defaults[self._indices]) * self.spec.joint_position_scale,
                v[self._indices] * self.spec.joint_velocity_scale,
                self._previous,
            )
        ).astype(np.float32)
        return frame

    def begin(
        self,
        *,
        joint_position: np.ndarray,
        joint_velocity: np.ndarray,
        projected_gravity: np.ndarray,
        angular_velocity: np.ndarray,
    ) -> None:
        self._check_valid()
        try:
            if self._started:
                raise ValueError("fresh episode entry cannot reset an existing history")
            frame = self._frame(joint_position, joint_velocity, projected_gravity, angular_velocity)
            self._history[:] = frame
            self._started = True
        except Exception:
            self._invalid = True
            raise

    def prepare(
        self,
        *,
        tick: int,
        joint_position: np.ndarray,
        joint_velocity: np.ndarray,
        projected_gravity: np.ndarray,
        angular_velocity: np.ndarray,
        command: np.ndarray,
    ) -> np.ndarray:
        self._check_valid()
        try:
            if (
                not self._started
                or self._pending
                or type(tick) is not int
                or tick != self._next_tick
                or not 0 <= tick < 10_000_000
            ):
                raise ValueError("consecutive completed AMP decision ticks required")
            wanted = _vector(command, 3, 10)
            if np.any(wanted < self._low) or np.any(wanted > self._high):
                raise ValueError("command exceeds declared AMP input range")
            alpha = np.float32(self.spec.command_smoothing)
            self._command = alpha * self._command + np.float32(
                1 - self.spec.command_smoothing
            ) * wanted.astype(np.float32)
            frame = self._frame(joint_position, joint_velocity, projected_gravity, angular_velocity)
            self._history[:-1] = self._history[1:].copy()
            self._history[-1] = frame
            self._pending = True
            return self._history.reshape(384).clip(-100, 100).copy()
        except Exception:
            self._invalid = True
            raise

    def commit_action(self, raw_action: np.ndarray) -> np.ndarray:
        """Commit one model output and return motor-order target proposals.

        Caller applies physical target/torque constraints and owns model errors.
        This is not an actuator call or a validation of the proposed trajectory.
        """
        self._check_valid()
        try:
            if not self._pending:
                raise ValueError("AMP action requires one pending observation")
            action = _vector(raw_action, 29, 100).astype(np.float32)
            target = np.empty(29, dtype=np.float32)
            target[self._indices] = (
                action * np.float32(self.spec.action_scale) + self._defaults[self._indices]
            )
            self._previous = action.copy()
            self._pending = False
            self._next_tick += 1
            return target
        except Exception:
            self._invalid = True
            raise

    def snapshot(self) -> AmpHistorySnapshot:
        self._check_valid()
        if not self._started or self._pending:
            raise ValueError("only a completed AMP decision boundary can be saved")
        return AmpHistorySnapshot(
            self.spec.spec_hash,
            self._next_tick,
            _readonly(self._history),
            _readonly(self._previous),
            _readonly(self._command),
        )

    def restore(self, snapshot: AmpHistorySnapshot) -> None:
        self._check_valid()
        try:
            if (
                self._started
                or not isinstance(snapshot, AmpHistorySnapshot)
                or snapshot.spec_hash != self.spec.spec_hash
                or type(snapshot.next_tick) is not int
                or not 0 <= snapshot.next_tick <= 10_000_000
                or not isinstance(snapshot.history, np.ndarray)
                or snapshot.history.shape != (4, 96)
                or snapshot.history.dtype != np.float32
                or not np.isfinite(snapshot.history).all()
                or np.any(np.abs(snapshot.history) > 10000)
            ):
                raise ValueError("complete matching AMP snapshot on a fresh instance required")
            previous = _vector(snapshot.previous_action, 29, 100).astype(np.float32)
            command = _vector(snapshot.smoothed_command, 3, 10).astype(np.float32)
            if (
                np.any(command < self._low)
                or np.any(command > self._high)
                or not np.array_equal(snapshot.history[-1, 6:9], command)
                or np.any(np.abs(np.linalg.norm(snapshot.history[:, 3:6], axis=1) - 1) > 1e-4)
                or np.any(np.abs(snapshot.history[:, 67:96]) > 100)
                or np.any(snapshot.history[:, 6:9] < self._low)
                or np.any(snapshot.history[:, 6:9] > self._high)
            ):
                raise ValueError("snapshot command differs from its last observation")
            self._history = snapshot.history.copy()
            self._previous = previous.copy()
            self._command = command.copy()
            self._next_tick = snapshot.next_tick
            self._started = True
        except Exception:
            self._invalid = True
            raise
