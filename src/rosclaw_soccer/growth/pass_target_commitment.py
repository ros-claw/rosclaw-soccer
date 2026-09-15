"""Bound a tactical receiver preference to a live, unlaunched peer agreement."""

from __future__ import annotations

from rosclaw_soccer.growth.pass_handoff import PassHandoff


def live_pass_flight(handoff: PassHandoff | None, *, time_sec: float) -> bool:
    """A proposal cannot replace an observed launch before its original expiry."""
    return bool(
        handoff is not None
        and not handoff.expired(time_sec)
        and handoff.source_foot_contact_sec is not None
    )


def preferred_unlaunched_receiver(
    handoff: PassHandoff | None,
    *,
    source: str,
    time_sec: float,
    possession_agent_id: str | None,
    source_ready: bool,
    receiver_ready: bool,
) -> str | None:
    if type(source_ready) is not bool or type(receiver_ready) is not bool:
        raise ValueError("observed motor readiness must be boolean")
    if handoff is None:
        return None
    expired = handoff.expired(time_sec)
    if (
        source != handoff.source
        or expired
        or handoff.source_foot_contact_sec is not None
        or possession_agent_id not in {None, source}
        or not source_ready
        or not receiver_ready
    ):
        return None
    return handoff.receiver
