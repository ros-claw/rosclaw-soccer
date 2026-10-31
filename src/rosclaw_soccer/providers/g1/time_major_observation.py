"""Simulation-only, causal G1 observation history for explicit policy contracts.

The caller supplies already transformed body-frame sensor measurements and a
body-frame velocity command. This class does not infer sensor frames, normalize
observations, run a model, drive a robot or qualify cross-simulator transfer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from rosclaw_soccer.providers.g1.joint_contract import G1_DDS_JOINT_NAMES


def _vector(value: tuple[float, ...], size: int, bound: float) -> tuple[float, ...]:
    if (
        type(value) is not tuple
        or len(value) != size
        or any(type(x) not in (int, float) or not math.isfinite(x) or abs(x) > bound for x in value)
    ):
        raise ValueError("finite bounded immutable observation vector required")
    return tuple(float(x) for x in value)


@dataclass(frozen=True)
class G1ProprioceptiveFrame:
    """One preprocessed observation; joint arrays use explicit DDS order."""

    angular_velocity: tuple[float, ...]
    projected_gravity: tuple[float, ...]
    command: tuple[float, ...]
    joint_position: tuple[float, ...]
    joint_velocity: tuple[float, ...]

    def __post_init__(self) -> None:
        for value, size, bound in (
            (self.angular_velocity, 3, 1000),
            (self.projected_gravity, 3, 1.001),
            (self.command, 3, 10),
            (self.joint_position, 29, 100),
            (self.joint_velocity, 29, 1000),
        ):
            _vector(value, size, bound)
        if abs(math.sqrt(sum(x * x for x in self.projected_gravity)) - 1.0) > 1e-4:
            raise ValueError("unit projected gravity required")


class G1TimeMajorObservation:
    """Four oldest-first 96-D frames, with exactly one action per decision.

    The first decision backfills the actual initial frame, including its
    command, four times with zero previous action. This matches MJLab's buffer
    initialization, but not every AMP model's history convention. For a new
    episode construct a new instance; never fabricate a mid-episode history.
    """

    activation_ceiling = "SIM_ONLY"

    def __init__(
        self,
        *,
        joint_names: tuple[str, ...],
        default_position: tuple[float, ...],
        default_velocity: tuple[float, ...],
    ) -> None:
        if type(joint_names) is not tuple or joint_names != G1_DDS_JOINT_NAMES:
            raise ValueError("exact G1 DDS joint order required")
        self._position = _vector(default_position, 29, 100)
        self._velocity = _vector(default_velocity, 29, 1000)
        self._history: tuple[tuple[float, ...], ...] = ()
        self._previous: tuple[float, ...] = (0.0,) * 29
        self._next_tick = 0
        self._pending = False
        self._faulted = False

    def prepare(self, *, tick: int, frame: G1ProprioceptiveFrame) -> tuple[float, ...]:
        if self._faulted:
            raise RuntimeError("observation history faulted")
        try:
            if type(tick) is not int or tick != self._next_tick or self._pending:
                raise ValueError("consecutive decision ticks with committed actions required")
            if not isinstance(frame, G1ProprioceptiveFrame):
                raise ValueError("typed sensor frame required")
            frame.__post_init__()
            current = (
                *frame.angular_velocity,
                *frame.projected_gravity,
                *frame.command,
                *(float(q - d) for q, d in zip(frame.joint_position, self._position, strict=True)),
                *(float(v - d) for v, d in zip(frame.joint_velocity, self._velocity, strict=True)),
                *self._previous,
            )
            assert len(current) == 96
            self._history = (current,) * 4 if not self._history else (*self._history[1:], current)
            self._pending = True
            return tuple(value for past in self._history for value in past)
        except Exception:
            self._faulted = True
            raise

    def commit_action(self, raw_action: tuple[float, ...]) -> None:
        if self._faulted:
            raise RuntimeError("observation history faulted")
        try:
            if not self._pending:
                raise ValueError("a prepared observation is required")
            self._previous = _vector(raw_action, 29, 100)
            self._pending = False
            self._next_tick += 1
        except Exception:
            self._faulted = True
            raise
