"""Morphology-independent, SIM-only event-triggered joint reference proposals.

This is an opt-in policy adapter, not an executor or a physical safety proof.
The caller supplies a measured event and applies the resulting proposal through
its existing joint/motor guards. Checkpoints and runtime defaults are unchanged.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from numbers import Real

import numpy as np

from rosclaw_soccer.sim.contracts import hash_json


def _finite_real(value: object) -> bool:
    if not isinstance(value, Real) or isinstance(value, (bool, np.bool_)):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


@dataclass(frozen=True)
class CausalReferencePulseConfig:
    amplitude_rad: tuple[float, ...]
    rise_sec: float
    hold_end_sec: float
    fade_sec: float
    maximum_combined_delta_rad: float
    activation_ceiling: str = "SIM_ONLY"

    def __post_init__(self) -> None:
        amplitudes = tuple(self.amplitude_rad)
        values = (*amplitudes, self.rise_sec, self.hold_end_sec, self.fade_sec)
        if not amplitudes or not all(_finite_real(value) for value in values):
            raise ValueError("reference pulse requires finite real amplitudes and times")
        if (
            not _finite_real(self.maximum_combined_delta_rad)
            or self.maximum_combined_delta_rad <= 0
        ):
            raise ValueError("reference pulse requires an explicit positive combined bound")
        if not (
            0 < self.rise_sec <= self.hold_end_sec
            and self.fade_sec > 0
            and math.isfinite(self.hold_end_sec + self.fade_sec)
        ):
            raise ValueError("reference pulse requires ordered rise, hold and end times")
        if max(abs(value) for value in amplitudes) > self.maximum_combined_delta_rad:
            raise ValueError("reference pulse amplitude exceeds the combined bound")
        if self.activation_ceiling != "SIM_ONLY":
            raise ValueError("reference pulse is SIM_ONLY")
        object.__setattr__(self, "amplitude_rad", tuple(float(value) for value in amplitudes))
        for name in ("rise_sec", "hold_end_sec", "fade_sec", "maximum_combined_delta_rad"):
            object.__setattr__(self, name, float(getattr(self, name)))

    @property
    def end_sec(self) -> float:
        return self.hold_end_sec + self.fade_sec

    @property
    def config_hash(self) -> str:
        return str(hash_json(asdict(self)))


@dataclass(frozen=True)
class CausalReferencePulseSample:
    delta_rad: np.ndarray
    combined_delta_rad: np.ndarray
    weight: float
    trigger_timestamp_sec: float | None


class CausalReferencePulse:
    """One episode/role, measured-event latch, no automatic rearm or fault reset."""

    def __init__(self, config: CausalReferencePulseConfig) -> None:
        self.config = config
        self._trigger_sec: float | None = None
        self._previous_sec: float | None = None
        self._faulted = False

    def sample(
        self,
        *,
        timestamp_sec: float,
        observed_trigger: bool,
        existing_delta_rad: np.ndarray,
    ) -> CausalReferencePulseSample:
        if self._faulted:
            raise ValueError("reference pulse is fault-latched; start a new episode")
        existing = np.asarray(existing_delta_rad)
        if (
            not _finite_real(timestamp_sec)
            or timestamp_sec < 0
            or (self._previous_sec is not None and timestamp_sec <= self._previous_sec)
            or type(observed_trigger) is not bool
            or existing.shape != (len(self.config.amplitude_rad),)
            or existing.dtype.kind not in "fiu"
            or not np.isfinite(existing).all()
        ):
            self._faulted = True
            raise ValueError("reference pulse requires ordered finite state and a boolean event")
        existing = existing.astype(np.float64)
        if np.max(np.abs(existing)) > self.config.maximum_combined_delta_rad + 1e-12:
            self._faulted = True
            raise ValueError("existing reference residual exceeds the combined bound")
        timestamp = float(timestamp_sec)
        if observed_trigger and self._trigger_sec is None:
            self._trigger_sec = timestamp
        weight = 0.0
        if self._trigger_sec is not None:
            elapsed = timestamp - self._trigger_sec
            if 0 <= elapsed < self.config.rise_sec:
                weight = math.sin(0.5 * math.pi * elapsed / self.config.rise_sec) ** 2
            elif self.config.rise_sec <= elapsed < self.config.hold_end_sec:
                weight = 1.0
            elif self.config.hold_end_sec <= elapsed < self.config.end_sec:
                weight = (
                    math.cos(
                        0.5 * math.pi * (elapsed - self.config.hold_end_sec) / self.config.fade_sec
                    )
                    ** 2
                )
        delta = np.asarray(self.config.amplitude_rad, dtype=np.float64) * weight
        combined = existing + delta
        if np.max(np.abs(combined)) > self.config.maximum_combined_delta_rad + 1e-12:
            self._faulted = True
            raise ValueError("combined reference residual exceeds its bound")
        self._previous_sec = timestamp
        delta.flags.writeable = False
        combined.flags.writeable = False
        return CausalReferencePulseSample(delta, combined, weight, self._trigger_sec)
