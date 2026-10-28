"""Causal SIM research gate for earlier whole-body receiving control."""

import numpy as np


def precontact_fullbody_gate(
    qpos: np.ndarray,
    qvel: np.ndarray,
    *,
    passer_foot_seen: bool,
    receiver_contact_seen: bool,
    world_healthy: bool,
) -> bool:
    """Trigger only on observed healthy approaching ball after actual pass.

    The gate proposes a motor transition for the next control. It performs no
    physics or actuation and is neither a reception nor a success certificate.
    """
    if (
        not isinstance(qpos, np.ndarray)
        or qpos.shape != (43,)
        or not isinstance(qvel, np.ndarray)
        or qvel.shape != (41,)
        or not np.isfinite(qpos).all()
        or not np.isfinite(qvel).all()
        or any(
            type(v) is not bool for v in (passer_foot_seen, receiver_contact_seen, world_healthy)
        )
    ):
        raise ValueError("finite canonical observed states and explicit past events required")
    delta = qpos[36:38] - qpos[:2]
    relative = qvel[35:37] - qvel[:2]
    distance = float(np.linalg.norm(delta))
    if distance == 0:
        return False
    closing = float(-np.dot(delta, relative) / distance)
    return bool(
        passer_foot_seen
        and not receiver_contact_seen
        and world_healthy
        and 0.4 <= distance <= 0.9
        and 0.5 <= closing <= 5.0
        and qpos[38] <= 0.3
        and qpos[2] >= 0.65
    )
