"""Explicit SIM-only held-gain protocol over one continuous native body."""

from dataclasses import dataclass
from typing import Any

import numpy as np

from rosclaw_soccer.physics.g1_target_episode import G1TargetEpisode
from rosclaw_soccer.sim.contracts import hash_json


@dataclass(frozen=True)
class G1GainSchedule:
    initial_kp: tuple[float, ...]
    initial_kd: tuple[float, ...]
    final_kp: tuple[float, ...]
    final_kd: tuple[float, ...]
    handoff_frame: int
    mode: str

    def __post_init__(self) -> None:
        if (
            self.mode not in ("constant", "linear10")
            or type(self.handoff_frame) is not int
            or not 30 <= self.handoff_frame <= 90
            or any(
                type(row) is not tuple
                or len(row) != 29
                or any(type(x) is not float or not np.isfinite(x) or not 0 < x <= 1000 for x in row)
                for row in (self.initial_kp, self.initial_kd, self.final_kp, self.final_kd)
            )
        ):
            raise ValueError("bounded immutable positive 29-joint SIM gain schedule required")

    def gains(self, frame: int) -> tuple[np.ndarray, np.ndarray]:
        if type(frame) is not int or not 0 <= frame < 300:
            raise ValueError("bounded current H300 frame required")
        kp, kd = np.array(self.initial_kp), np.array(self.initial_kd)
        if self.mode == "constant" or frame < self.handoff_frame:
            return kp, kd
        alpha = min(1.0, (frame - self.handoff_frame + 1) / 10)
        if alpha == 1:
            return np.array(self.final_kp), np.array(self.final_kd)
        return (
            kp + alpha * (np.array(self.final_kp) - kp),
            kd + alpha * (np.array(self.final_kd) - kd),
        )

    def report(self) -> dict[str, Any]:
        return dict(
            schema="soccer.physics.g1_explicit_gain_schedule.v1",
            **self.__dict__,
            intervals=10,
            first_interval_alpha=0.1,
            gain_changes_require_actual_handoff=True,
            physical_body_reset=False,
            hardware_authorized=False,
            promotion_authorized=False,
        )

    @property
    def contract_hash(self) -> str:
        return hash_json(self.report())


class G1GainScheduledEpisode(G1TargetEpisode):
    """Never resets data/time/warmstart; changes only declared controller gains.

    No second integrator: each target still uses the unchanged G1TargetEpisode
    ten native steps and hard torque limits. Gain records are held per 50Hz tick.
    """

    def __init__(self, model: Any, *, schedule: G1GainSchedule, **kwargs: Any) -> None:
        schedule.__post_init__()
        kp, kd = schedule.gains(0)
        initial_kp, initial_kd = kwargs.get("kp"), kwargs.get("kd")
        if (
            not isinstance(initial_kp, np.ndarray)
            or not isinstance(initial_kd, np.ndarray)
            or not np.array_equal(initial_kp, kp)
            or not np.array_equal(initial_kd, kd)
        ):
            raise ValueError("constructor must start at exact declared gains")
        self.schedule = schedule
        self.transition_activated = False
        super().__init__(model, **kwargs)

    def activate(self, frame: int) -> None:
        if (
            type(frame) is not int
            or frame != self.schedule.handoff_frame
            or frame != self.next_frame
            or self.failed
            or self.transition_activated
        ):
            raise ValueError("exact once actual current handoff required")
        self.transition_activated = True

    def step(self, frame: int, target: np.ndarray) -> dict[str, np.ndarray]:
        if (
            type(frame) is not int
            or frame != self.next_frame
            or self.failed
            or (frame >= self.schedule.handoff_frame and not self.transition_activated)
        ):
            raise ValueError("sequential explicit SIM schedule and actual handoff required")
        self.kp, self.kd = self.schedule.gains(frame)
        row = super().step(frame, target)
        row.update(held_pd_kp=self.kp[None].copy(), held_pd_kd=self.kd[None].copy())
        return row
