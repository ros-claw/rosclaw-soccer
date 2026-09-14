"""Causal kick-follow-through timing proposals for simulation experiments.

This helper neither detects contact nor executes motion. Callers must supply
the first measured foot-ball contact tick, retain physical safety checks, and
separately qualify every controller transition. A contact is not a goal.
"""

from __future__ import annotations

from dataclasses import dataclass


def _integer(value: int, name: str, minimum: int = 0) -> None:
    if type(value) is not int or not minimum <= value <= 10**9:
        raise ValueError(f"{name} must be a bounded integer >= {minimum}")


@dataclass(frozen=True)
class KickRecoveryTimingConfig:
    kick_start_frame: int
    latest_recovery_frame: int
    followthrough_frames: int = 10
    blend_frames: int = 10
    physics_ticks_per_frame: int = 10

    def __post_init__(self) -> None:
        _integer(self.kick_start_frame, "kick_start_frame")
        _integer(self.latest_recovery_frame, "latest_recovery_frame")
        _integer(self.followthrough_frames, "followthrough_frames")
        _integer(self.blend_frames, "blend_frames", 1)
        _integer(self.physics_ticks_per_frame, "physics_ticks_per_frame", 1)
        if self.latest_recovery_frame < self.kick_start_frame:
            raise ValueError("recovery deadline cannot precede kick entry")


@dataclass(frozen=True)
class KickRecoveryTimingDecision:
    contact_observed_frame: int | None
    recovery_start_frame: int
    recovery_blend: float


class KickRecoveryTiming:
    """One episode's monotonic timing state; construct anew for each episode.

    Tick indices are zero-based post-step samples. At control frame F, only
    ticks < F * physics_ticks_per_frame have been observed. The provided tick
    must be the first contact since kick entry, not a later replacement.
    Invalid input raises before changing state; this is not a runtime fallback.
    """

    def __init__(self, config: KickRecoveryTimingConfig) -> None:
        if not isinstance(config, KickRecoveryTimingConfig):
            raise ValueError("validated kick timing config required")
        self.config = config
        self._last_frame = -1
        self._first_tick: int | None = None

    def advance(
        self, control_frame: int, *, first_contact_tick: int | None
    ) -> KickRecoveryTimingDecision:
        _integer(control_frame, "control_frame")
        if control_frame <= self._last_frame:
            raise ValueError("control frames must strictly increase")
        cfg = self.config
        if first_contact_tick is not None:
            _integer(first_contact_tick, "first_contact_tick")
            if not (
                cfg.kick_start_frame * cfg.physics_ticks_per_frame
                <= first_contact_tick
                < control_frame * cfg.physics_ticks_per_frame
            ):
                raise ValueError("contact must be observed and since kick entry")
            if self._first_tick is not None and first_contact_tick != self._first_tick:
                raise ValueError("first contact cannot be rewritten")
        tick = self._first_tick if self._first_tick is not None else first_contact_tick
        observed_frame = tick // cfg.physics_ticks_per_frame + 1 if tick is not None else None
        start = cfg.latest_recovery_frame
        if observed_frame is not None:
            start = min(start, observed_frame + cfg.followthrough_frames)
        blend = min(1.0, max(0.0, (control_frame - start + 1) / cfg.blend_frames))
        self._first_tick = tick
        self._last_frame = control_frame
        return KickRecoveryTimingDecision(observed_frame, start, blend)
