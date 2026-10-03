"""Explicit MuJoCo root velocity reference points, without refreshing physics.

BODY measures velocity at the inertial COM; XBODY at the regular body origin.
Both read the same cached dynamics stage. Neither is a current-qvel Jacobian
refresh after integration. Keep that temporal boundary explicit in evidence.
"""

from typing import Any

import numpy as np


def root_observation_contract(reference: str) -> dict[str, str] | None:
    if reference == "body-com":
        return None  # Exact historical commitment remains unchanged.
    if reference != "body-origin":
        raise ValueError("unknown root velocity reference")
    return {
        "root_velocity_reference": "body-origin",
        "velocity_order": "world-linear-then-world-angular",
        "derived_state_stage": "cached-last-forward-stage-no-refresh",
    }


def reference_from_contract(contract: Any) -> str:
    if contract is None:
        return "body-com"
    if type(contract) is not dict or contract != root_observation_contract("body-origin"):
        raise ValueError("unsupported root observation contract")
    return "body-origin"


def root_velocity_world(model: Any, data: Any, body_id: int, reference: str) -> np.ndarray:
    """Read world [linear, angular] velocity without step/forward or mutation."""
    import mujoco

    root_observation_contract(reference)
    if type(body_id) is not int or not 0 < body_id < model.nbody:
        raise ValueError("a non-world body id is required")
    kind = mujoco.mjtObj.mjOBJ_BODY if reference == "body-com" else mujoco.mjtObj.mjOBJ_XBODY
    out = np.zeros(6)
    mujoco.mj_objectVelocity(model, data, kind, body_id, out, 0)
    if not np.isfinite(out).all():
        raise ValueError("nonfinite root velocity")
    return out[[3, 4, 5, 0, 1, 2]]
