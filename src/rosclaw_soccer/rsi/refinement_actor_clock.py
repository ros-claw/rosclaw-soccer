"""Versioned actual actor clocks; no physics, padding, or legacy relabelling."""

from typing import Any

LEGACY = "pure-refinement-stochastic-native-v1"
FULL = "pure-refinement-full-recovery-stochastic-native-v2"


def validate_actor_clock(episode: dict[str, Any], n: int) -> bool:
    """Return full-coverage mode after checking the observed terminal clock.

    The formal motion deadline remains 300; at most 100 recovery controls
    follow. A failed episode stops at its actual last control in either phase.
    """
    kind = episode.get("controller_kind")
    full = kind == FULL
    motion = episode.get("motion_reference_controls")
    tail = episode.get("observed_recovery_controls")
    total = episode.get("total_observed_controls")
    terminal = episode.get("termination")
    if (
        kind not in (LEGACY, FULL)
        or type(n) is not int
        or not 1 <= n <= (400 if full else 300)
        or type(motion) is not int
        or not 1 <= motion <= 300
        or type(tail) is not int
        or not 0 <= tail <= 100
        or type(total) is not int
        or total != (n if full else n + tail)
        or episode.get("deterministic_recovery_excluded_from_actor_rows") is not (not full)
        or terminal not in ("BODY_FAILURE", "FULL_MOTION_PLUS100NN_RECOVERY")
    ):
        raise ValueError("actual versioned terminal recovery clock required")
    observed_motion = n - tail if full else n
    if (
        not 1 <= observed_motion <= motion
        or (tail > 0 and observed_motion != motion)
        or (
            terminal == "FULL_MOTION_PLUS100NN_RECOVERY"
            and (observed_motion != motion or tail != 100)
        )
    ):
        raise ValueError("actual terminal recovery clock required; failures cannot be padded")
    return full
