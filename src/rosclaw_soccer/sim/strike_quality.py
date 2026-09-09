"""Robot-independent, physics-rate Soccer strike evidence accounting.

This consumes measured forces and motion, never actuates or promotes a policy.
Callers own named geometry groups and must exclude only declared floor/support
contacts. A reference phase label alone is not proof of an intentional shot.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

from rosclaw_soccer.sim.contracts import hash_json


@dataclass(frozen=True)
class StrikeQualityConfig:
    physics_dt_sec: float = 0.002
    duration_sec: float = 4.0
    strike_reference_start_sec: float = 0.0
    force_threshold_n: float = 1.0
    attribution_window_sec: float = 0.2
    directed_ball_threshold_mps: float = 3.0
    forward_speed_floor_mps: float = 0.1

    def __post_init__(self) -> None:
        values = asdict(self)
        if any(type(v) not in (float, int) or not math.isfinite(v) for v in values.values()):
            raise ValueError("finite numeric strike-quality configuration required")
        if (
            not 0 < self.physics_dt_sec <= 0.01
            or not self.physics_dt_sec <= self.duration_sec <= 60
            or not 0 <= self.strike_reference_start_sec < self.duration_sec
            or not 0 < self.force_threshold_n <= 100
            or not self.physics_dt_sec <= self.attribution_window_sec <= 0.5
            or not 0 < self.directed_ball_threshold_mps <= 30
            or not 0 <= self.forward_speed_floor_mps <= 5
            or not math.isclose(
                self.duration_sec / self.physics_dt_sec,
                round(self.duration_sec / self.physics_dt_sec),
                abs_tol=1e-8,
            )
        ):
            raise ValueError("bounded physics-rate strike-quality contract required")

    @property
    def contract_hash(self) -> str:
        return str(hash_json(asdict(self)))


class StrikeQualityTracker:
    """One declared handoff episode, consecutive samples, no silent recovery."""

    def __init__(self, config: StrikeQualityConfig | None = None) -> None:
        self.config = config or StrikeQualityConfig()
        self._samples = 0
        self._faulted = False
        self._safe = True
        self._blocked = False
        self._first_foot: float | None = None
        self._first_other: float | None = None
        self._minimum_forward = math.inf
        self._peak = 0.0

    def observe(
        self,
        *,
        elapsed_sec: float,
        body_safe: bool,
        forward_speed_mps: float,
        directed_ball_speed_mps: float,
        foot_normal_force_n: float,
        other_non_ground_normal_force_n: float,
    ) -> None:
        if self._faulted:
            raise ValueError("strike evidence is fault-latched; start a new declared episode")
        values = (
            elapsed_sec,
            forward_speed_mps,
            directed_ball_speed_mps,
            foot_normal_force_n,
            other_non_ground_normal_force_n,
        )
        expected = (self._samples + 1) * self.config.physics_dt_sec
        if (
            type(body_safe) is not bool
            or any(type(v) not in (float, int) or not math.isfinite(v) for v in values)
            or not math.isclose(elapsed_sec, expected, rel_tol=0, abs_tol=1e-8)
            or elapsed_sec > self.config.duration_sec + 1e-8
            or foot_normal_force_n < 0
            or other_non_ground_normal_force_n < 0
        ):
            self._faulted = True
            raise ValueError("invalid, missing, duplicate or nonfinite physics sample")
        self._samples += 1
        self._safe &= body_safe
        if self._first_foot is None:
            self._minimum_forward = min(self._minimum_forward, forward_speed_mps)
        if other_non_ground_normal_force_n > self.config.force_threshold_n:
            self._blocked = True
            if self._first_other is None:
                self._first_other = elapsed_sec
        if foot_normal_force_n > self.config.force_threshold_n and self._first_foot is None:
            self._first_foot = elapsed_sec
        if (
            self._first_foot is not None
            and self._safe
            and not self._blocked
            and elapsed_sec - self._first_foot <= self.config.attribution_window_sec + 1e-8
        ):
            self._peak = max(self._peak, directed_ball_speed_mps)

    def result(self) -> dict[str, object]:
        complete = self._samples == round(self.config.duration_sec / self.config.physics_dt_sec)
        admitted = complete and not self._faulted and self._safe
        timing = self._first_foot is not None and (
            self._first_foot + 1e-8 >= self.config.strike_reference_start_sec
        )
        forward = self._first_foot is not None and (
            self._minimum_forward > self.config.forward_speed_floor_mps
        )
        return {
            "schema": "rosclaw_soccer.strike_quality.v1",
            "config_hash": self.config.contract_hash,
            "samples": self._samples,
            "complete": complete,
            "faulted": self._faulted,
            "body_safe_entire_episode": self._safe,
            "first_foot_sec": self._first_foot,
            "first_other_sec": self._first_other,
            "minimum_precontact_forward_mps": (
                self._minimum_forward if math.isfinite(self._minimum_forward) else None
            ),
            "clean_directed_peak_mps": self._peak,
            "clean_touch": admitted and self._peak > 1.0,
            "reference_strike_phase_contact": timing,
            "uninterrupted_forward": forward,
            "strong_forward_strike": (
                admitted
                and timing
                and forward
                and self._peak > self.config.directed_ball_threshold_mps
            ),
            "activation_ceiling": "SIM_ONLY",
            "promotion_eligible": False,
        }
