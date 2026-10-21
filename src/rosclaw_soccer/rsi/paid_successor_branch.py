"""Explicit TRAIN-only counterfactual branch from an audited physical entry.

State restoration is declared here and MUST NOT be presented as an uninterrupted
match or a newly learned autonomous state. No source evidence is overwritten.
"""

from typing import Any

import numpy as np

from rosclaw_soccer.rsi.paid_continuity_audit import identical, require_integration_fields
from rosclaw_soccer.rsi.paid_skill_handoff import snapshot


def begin_successor_branch(
    bench: Any, case: dict[str, Any], entry: dict[str, np.ndarray], previous_action: np.ndarray
) -> Any:
    import mujoco

    require_integration_fields(entry)
    if (
        previous_action.shape != (29,)
        or previous_action.dtype != np.float32
        or not np.isfinite(previous_action).all()
    ):
        raise ValueError("actual private preceding29 action required")
    ep = bench.make(case)
    d = ep.env.data
    if any(getattr(d, key).shape != value.shape for key, value in entry.items() if key != "time"):
        raise ValueError("exact same physical model state shapes required")
    for key, value in entry.items():
        if key == "time":
            d.time = float(value)
        else:
            getattr(d, key)[:] = value
    identical(snapshot(d), entry)
    # A new mjData still has contacts/kinematics from the factory's initial
    # state. Rebuild those derived fields before admission at the restored
    # actual state; retain the original recorded integration warmstart.
    mujoco.mj_forward(ep.env.model, d)
    d.qacc_warmstart[:] = entry["qacc_warmstart"]
    identical(snapshot(d), entry)
    ep.done = True
    ep.termination_reason = "REFERENCE_COMPLETE"
    ep.last_action = previous_action.copy()
    ep.begin_current_state(ep.clip, goal=entry["mocap_pos"][ep.env.goal_mocap_id].copy())
    identical(snapshot(d), entry)
    if not np.array_equal(ep.last_action, previous_action) or ep.frame != 0 or ep.done:
        raise ValueError("explicit branch changed its skill boundary/action")
    return ep
