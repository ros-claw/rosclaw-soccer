"""Combine existing capture evidence with a sampled match-boundary veto.

This does not change any legacy reward or upgrade a legacy outcome. It is
not a teacher qualification: ball positions are sampled at 50 Hz, the contact
recorder omits ground, and team intent and full safety are separate evidence.
"""

from typing import Any

from rosclaw_soccer.training.substep_capture_audit import audit_substep_capture
from rosclaw_soccer.world.match_boundary import uninterrupted_in_play


def audit_in_play_substep_capture(
    trace: dict[str, Any], *, agent_ids: tuple[str, ...], agent_id: str, start: int, frames: int
) -> dict[str, Any]:
    """Veto a capture if any sampled match exit occurs before its window ends.

    An earlier exit remains disqualifying even if the ball later re-enters:
    this contract has no independently authenticated restart. A first exit
    after the window is retained as context but does not rewrite its outcome.
    Missing or malformed boundary or substep evidence fails closed.
    """
    capture = audit_substep_capture(
        trace, agent_ids=agent_ids, agent_id=agent_id, start=start, frames=frames
    )
    boundary = uninterrupted_in_play(trace)
    first_exit = boundary["first_exit_frame"]
    sampled_in_play = first_exit is None or first_exit >= start + frames
    return {
        "schema": "soccer.in_play_substep_capture_audit.v1",
        "capture": capture,
        "boundary": boundary,
        "sampled_in_play_through_window": sampled_in_play,
        "sampled_in_play_capture_without_recorded_substep_veto": bool(
            sampled_in_play and capture["capture_without_recorded_substep_veto"]
        ),
        "continuous_boundary_verified": False,
        "team_intent_verified": False,
        "training_authorized": False,
        "promotion_eligible": False,
        "activation_ceiling": "SIM_ONLY",
    }
