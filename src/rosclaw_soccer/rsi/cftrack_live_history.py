"""Capture causal, owned MuJoCo boundaries for experimental CFtrack transfer."""

from collections import deque
from dataclasses import dataclass

import numpy as np

from rosclaw_soccer.rsi.shared_native_motor_world import SharedNativeMotorWorld


@dataclass(frozen=True)
class CFtrackBoundarySample:
    control: int
    time_sec: float
    canonical_qpos: tuple[float, ...]
    canonical_qvel: tuple[float, ...]
    cached_ball_position: tuple[float, ...]
    cached_ball_quaternion: tuple[float, ...]
    previous_executed_target: tuple[float, ...]


class CFtrackLiveHistory:
    """One owner, one player and five actual consecutive committed boundaries.

    This is only an observation collector. It does not create, enter or modify
    a policy, change gains, advance the world or authorize a skill transition.
    """

    def __init__(self, world: SharedNativeMotorWorld, prefix: str) -> None:
        if type(world) is not SharedNativeMotorWorld or prefix not in world.layouts:
            raise ValueError("one owned shared SIM world and player required")
        world._require_current_boundary()
        self._world, self._prefix = world, prefix
        self._samples: deque[CFtrackBoundarySample] = deque(maxlen=5)
        self._failed = False

    def capture(self) -> CFtrackBoundarySample:
        """Capture once after each real control; gaps and repeats poison this collector."""
        w = self._world
        try:
            w._require_current_boundary()
            if self._failed or w.control_steps < 1:
                raise ValueError("collector must be live after a committed control")
            if self._samples and w.control_steps != self._samples[-1].control + 1:
                raise ValueError("consecutive, nonrepeated physical boundaries required")
            lo = w.layouts[self._prefix]
            q, v = lo.canonical(w.data)
            target = w.last_applied_target(self._prefix)
            ball_position = w.data.xpos[lo.ball].copy()
            ball_quaternion = w.data.xquat[lo.ball].copy()
            if (
                not np.isfinite(w.data.time)
                or not np.isfinite(ball_position).all()
                or not np.isfinite(ball_quaternion).all()
                or abs(np.linalg.norm(ball_quaternion) - 1.0) > 1e-6
                or (self._samples and abs(w.data.time - self._samples[-1].time_sec - 0.02) > 1e-8)
            ):
                raise ValueError("finite actual cached ball and consecutive 50Hz time required")
            sample = CFtrackBoundarySample(
                control=w.control_steps,
                time_sec=float(w.data.time),
                canonical_qpos=tuple(map(float, q)),
                canonical_qvel=tuple(map(float, v)),
                cached_ball_position=tuple(map(float, ball_position)),
                cached_ball_quaternion=tuple(map(float, ball_quaternion)),
                previous_executed_target=tuple(map(float, target)),
            )
            self._samples.append(sample)
            return sample
        except BaseException:
            self._failed = True
            raise

    def at_current_boundary(self) -> tuple[CFtrackBoundarySample, ...]:
        """Return five samples only if the final sample still matches the live world."""
        w = self._world
        try:
            w._require_current_boundary()
            if self._failed or len(self._samples) != 5:
                raise ValueError("five consecutive actual samples required")
            sample = self._samples[-1]
            lo = w.layouts[self._prefix]
            q, v = lo.canonical(w.data)
            if (
                sample.control != w.control_steps
                or sample.time_sec != float(w.data.time)
                or not np.array_equal(q, sample.canonical_qpos)
                or not np.array_equal(v, sample.canonical_qvel)
                or not np.array_equal(w.data.xpos[lo.ball], sample.cached_ball_position)
                or not np.array_equal(w.data.xquat[lo.ball], sample.cached_ball_quaternion)
                or not np.array_equal(
                    w.last_applied_target(self._prefix), sample.previous_executed_target
                )
            ):
                raise ValueError(
                    "current physical state, cache and executed target must still match"
                )
            return tuple(self._samples)
        except BaseException:
            self._failed = True
            raise
