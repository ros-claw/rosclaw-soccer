"""Detect a silent scientific queue; never impersonate an agent wakeup service."""

import math
from typing import Any


def continuity_state(
    dispatch: dict[str, Any],
    previous: dict[str, Any] | None,
    *,
    now: float,
    successor_deadline_seconds: float = 300,
) -> dict[str, Any]:
    """Keep an idle deadline across observer ticks, including artifact changes.

    A timer heartbeat is not a successor. Only actually observed live declared
    work clears this timer. Wall-clock rollback is explicit and cannot satisfy
    an expired deadline. This report never launches work or approves retries.
    """
    if (
        type(now) not in (int, float)
        or not math.isfinite(now)
        or type(successor_deadline_seconds) not in (int, float)
        or not math.isfinite(successor_deadline_seconds)
        or successor_deadline_seconds <= 0
        or type(dispatch.get("no_live_declared_work")) is not bool
        or type(dispatch.get("live_units")) is not list
        or dispatch["no_live_declared_work"] != (len(dispatch["live_units"]) == 0)
    ):
        raise ValueError("finite continuity clock and consistent observed dispatch required")
    previous = previous or {}
    old_time = previous.get("observed_unix")
    old_since = previous.get("no_live_since_unix")
    for value in (old_time, old_since):
        if value is not None and (type(value) not in (int, float) or not math.isfinite(value)):
            raise ValueError("finite prior continuity timestamps required")
    if old_since is not None and (old_time is None or old_since > old_time):
        raise ValueError("ordered prior idle observation timestamps required")
    rollback = old_time is not None and now < old_time
    idle = dispatch["no_live_declared_work"]
    since = min(now, old_since) if idle and old_since is not None else now if idle else None
    age = 0.0 if since is None else now - since
    overdue = idle and age >= successor_deadline_seconds
    state = (
        "CLOCK_REVIEW_REQUIRED"
        if rollback
        else "SCIENTIFIC_SUCCESSOR_OVERDUE"
        if overdue
        else "SCIENTIFIC_SUCCESSOR_REQUIRED"
        if idle
        else "DECLARED_WORK_OBSERVED_LIVE"
    )
    return dict(
        schema="soccer.research.loop_continuity.v1",
        state=state,
        observed_unix=now,
        no_live_since_unix=since,
        idle_seconds=age,
        successor_deadline_seconds=successor_deadline_seconds,
        successor_overdue=overdue,
        wall_clock_rollback=rollback,
        agent_attention_required=idle
        or rollback
        or dispatch.get("agent_attention_required", False),
        observer_tick_is_scientific_progress=False,
        agent_wakeup_provided=False,
        automatic_physics_retry=False,
        goal_complete=False,
    )
