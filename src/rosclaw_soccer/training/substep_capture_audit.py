"""Conservative 500 Hz contact veto for an unchanged sampled capture exam.

The legacy 50 Hz outcome is never upgraded. Whole 20 ms bins are treated as
ambiguous, because contact ordering inside a bin is not retained. This is a
SIM-only evidence audit, not a motor command or a promotion decision.
"""

from typing import Any

import numpy as np

from rosclaw_soccer.training.receiving_rollout import receiving_window


def audit_substep_capture(
    trace: dict[str, Any], *, agent_ids: tuple[str, ...], agent_id: str, start: int, frames: int
) -> dict[str, Any]:
    """Veto sampled success when non-foot/foreign-foot contact was missed.

    Inputs must have been recorded for every 2 ms physics step of the 50 Hz
    trace. A missing or malformed stream raises instead of assuming clean.
    This cannot recover missed own-foot touches or establish full contact
    forces; it only adds evidence against a sampled positive result.
    """
    _, sampled = receiving_window(
        trace, agent_ids=agent_ids, agent_id=agent_id, start=start, frames=frames
    )
    n = len(trace["time"])
    own = agent_ids.index(agent_id)
    foot = np.asarray(trace["contact_normal_substeps"])
    nonfoot = np.asarray(trace["contact_nonfoot_substeps"])
    safe = np.asarray(trace["contact_normal_world_safe"])
    if (
        foot.shape != (n, len(agent_ids))
        or nonfoot.shape != (n,)
        or safe.shape != (n,)
        or foot.dtype.kind not in "iu"
        or nonfoot.dtype.kind not in "iu"
        or safe.dtype != np.bool_
        or np.any((foot < 0) | (foot > 10))
        or np.any((nonfoot < 0) | (nonfoot > 10))
    ):
        raise ValueError("complete bounded substep contact and safety coverage required")
    t = np.asarray(trace["time"], dtype=float)
    if not np.allclose(np.diff(t), 0.02, atol=1e-8, rtol=0):
        raise ValueError("contiguous physical 50 Hz control clock required")
    sampled_time = sampled["first_foot_contact_sec"]
    if sampled_time is None:
        return {
            "schema": "soccer.substep_capture_audit.v1",
            "sampled": sampled,
            "substep_status": "NO_SAMPLED_TOUCH",
            "capture_without_recorded_substep_veto": False,
            "promotion_eligible": False,
            "activation_ceiling": "SIM_ONLY",
        }
    first = int(np.searchsorted(t, sampled_time - 1e-9))
    end = start + frames
    if first < start or first >= end or foot[first, own] == 0:
        raise ValueError("sampled touch is inconsistent with complete substep evidence")
    forbidden_foot = np.any(np.delete(foot[first:end], own, axis=1) > 0)
    forbidden_nonfoot = np.any(nonfoot[first:end] > 0)
    unsafe = not bool(np.all(safe[first:end]))
    status = (
        "NONFOOT_CONTACT"
        if forbidden_nonfoot
        else "OTHER_FOOT_CONTACT"
        if forbidden_foot
        else "SUBSTEP_WORLD_UNSAFE"
        if unsafe
        else "NO_SUBSTEP_VETO"
    )
    return {
        "schema": "soccer.substep_capture_audit.v1",
        "sampled": sampled,
        "substep_status": status,
        "capture_without_recorded_substep_veto": bool(
            sampled["controlled_reception"] and status == "NO_SUBSTEP_VETO"
        ),
        "promotion_eligible": False,
        "activation_ceiling": "SIM_ONLY",
    }
