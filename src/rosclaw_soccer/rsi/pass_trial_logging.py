"""Separate coarse progress from private sealed case rows; never score video/logs."""

from typing import Any


def trial_progress(result: dict[str, Any], *, include_course: bool) -> dict[str, Any]:
    if type(include_course) is not bool:
        raise ValueError("explicit physical course log disclosure mode required")
    if include_course:
        return {k: v for k, v in result.items() if k != "artifacts"}
    # No coordinate, latent, outcome, state/goal ID, or nested case data.
    return {k: result[k] for k in ("index", "controls", "native_steps", "actor_calls")}
