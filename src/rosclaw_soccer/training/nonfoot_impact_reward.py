"""Bounded 50 Hz nonfoot impact training credit, never a safety certificate."""

from typing import Any

import numpy as np
from numpy.typing import NDArray

from rosclaw_soccer.growth.role_self_model import TacticalIntent


def nonfoot_impact_adjustment(
    trace: dict[str, Any], agent_ids: tuple[str, ...]
) -> NDArray[np.float64]:
    """Penalize sampled, attributed high-force nonfoot impacts during ball play.

    The recorded force is the peak over a control frame, not an impulse. Ball
    velocity change covers the entire frame and is used only when no sampled
    foot contact overlaps. The logger retains one strongest body contact, so
    actor attribution and cause are approximate; multiple simultaneous body
    contacts may be hidden. No contact is promoted to a physical success.
    Goalkeepers and SAVE/DISTRIBUTE are deliberately excluded: hands and
    bodies are legitimate for a keeper. Unlisted future tactical intents,
    such as a dedicated heading skill, are excluded by the allowlist too.
    """
    count = len(trace["time"])
    codes = np.asarray(trace["ball_nonfoot_contact_agent_code"])
    force = np.asarray(trace["ball_nonfoot_contact_force_n"], dtype=np.float64)
    foot_force = np.asarray(trace["ball_contact_force_n"], dtype=np.float64)
    before = np.asarray(trace["ball_control_entry_velocity"], dtype=np.float64)
    after = np.asarray(trace["ball_velocity"], dtype=np.float64)
    if (
        count == 0
        or len(agent_ids) != 8
        or tuple(sorted(set(agent_ids))) != agent_ids
        or codes.shape != (count,)
        or codes.dtype.kind not in "iu"
        or not np.isin(codes, range(9)).all()
        or force.shape != (count,)
        or foot_force.shape != (count,)
        or before.shape != (count, 6)
        or after.shape != (count, 6)
        or not all(np.isfinite(a).all() for a in (force, foot_force, before, after))
        or np.any(force < 0)
        or np.any(foot_force < 0)
        or np.any((force > 1.0) & (codes == 0))
    ):
        raise ValueError("aligned finite attributed contact and velocity evidence required")
    allowed = [
        list(TacticalIntent).index(intent)
        for intent in (
            TacticalIntent.RECEIVE,
            TacticalIntent.INTERCEPT,
            TacticalIntent.PRESS,
            TacticalIntent.CARRY,
            TacticalIntent.PASS,
            TacticalIntent.SHOOT,
        )
    ]
    delta_speed = np.linalg.norm(after[:, :3] - before[:, :3], axis=1)
    cost = 0.15 * np.minimum(force / 100.0, 1.0)
    cost += 0.25 * np.minimum(delta_speed / 2.0, 1.0) * (foot_force <= 1.0)
    result = np.zeros((count, len(agent_ids)), dtype=np.float64)
    for column, agent in enumerate(agent_ids):
        intent = np.asarray(trace[agent.replace(".", "_") + "_intent_code"])
        if (
            intent.shape != (count,)
            or intent.dtype.kind not in "iu"
            or not np.isin(intent, range(len(TacticalIntent))).all()
        ):
            raise ValueError("valid observed tactical intent required for every actor")
        if agent.rsplit(".", 1)[-1] == "goalkeeper":
            continue
        active = (codes == column + 1) & (force > 1.0) & np.isin(intent, allowed)
        result[:, column] = -np.where(active, cost, 0.0)
    return result
