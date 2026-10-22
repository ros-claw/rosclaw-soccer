"""Expose idle scientific queues explicitly; this is not an agent wakeup source."""

from typing import Any


def dispatch_state(campaigns: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Require agent work after termination, not a claim that a timer is training."""
    if not campaigns or len(campaigns) > 128:
        raise ValueError("bounded nonempty campaign map required")
    live, review, ready = [], [], []
    for unit, observation in sorted(campaigns.items()):
        phase = observation.get("phase")
        if phase in ("RUNNING", "COMPUTING_WITHOUT_NEW_COMMIT", "QUEUED_DECLARED_JOB"):
            live.append(unit)
        elif phase == "STAGE_ARTIFACT_READY_FOR_READBACK":
            ready.append(unit)
        else:
            # Includes hangs, malformed observations, failed and unloaded jobs.
            review.append(unit)
    if review:
        state, action = "AGENT_REVIEW_REQUIRED", "DIAGNOSE_PRESERVE_NO_BLIND_RETRY"
    elif ready:
        state, action = "AGENT_SUCCESSOR_REQUIRED", "AUTHENTICATE_RESULTS_DECLARE_NEXT_STAGE"
    else:
        state, action = "DECLARED_WORK_LIVE", "DEVELOP_NEXT_STAGE_WHILE_WORK_EXECUTES"
    return dict(
        state=state,
        next_agent_action=action,
        live_units=live,
        review_units=review,
        readback_units=ready,
        no_live_declared_work=not live,
        agent_attention_required=bool(review or ready),
        agent_wakeup_provided=False,
        scientific_successor_automated=False,
        physics_retry_authorized=False,
        goal_complete=False,
    )
