"""Report observed MuJoCo ball-ground contact dimensions, never calibration."""

from dataclasses import dataclass

import mujoco
import numpy as np


@dataclass(frozen=True)
class NativeBallGroundContact:
    status: str
    effective_dimensions: tuple[int, ...]
    effective_rolling_friction_m: tuple[tuple[float, float] | None, ...]
    inactive_contacts: int = 0
    hardware_authorized: bool = False


def inspect_ball_ground_contact(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    ball_geom: int,
    ground_geom: int,
) -> NativeBallGroundContact:
    """Read actual contact entries, including overrides from explicit pairs.

    Caller must supply matching model/data after forward at its current state.
    Contact entries remaining after a step can describe the pre-integration
    stage; this read does not infer or verify when they were last refreshed.
    No contact is unknown, not evidence of zero friction. Dimension six and
    nonzero coefficients indicate available rolling constraints, not that a
    particular pitch is calibrated or that energy is actually dissipated.
    This function does not modify simulator state or authorize any motion.
    """
    if not isinstance(model, mujoco.MjModel) or not isinstance(data, mujoco.MjData):
        raise TypeError("MuJoCo model and data required")
    if (
        type(ball_geom) is not int
        or type(ground_geom) is not int
        or not 0 <= ball_geom < model.ngeom
        or not 0 <= ground_geom < model.ngeom
        or ball_geom == ground_geom
        or data.qpos.shape != (model.nq,)
        or data.qvel.shape != (model.nv,)
        or not np.isfinite(data.qpos).all()
        or not np.isfinite(data.qvel).all()
    ):
        raise ValueError("distinct valid geoms and finite compatible simulation state required")
    dimensions: list[int] = []
    rolling: list[tuple[float, float] | None] = []
    inactive = 0
    for contact in data.contact:
        if set(map(int, contact.geom)) != {ball_geom, ground_geom}:
            continue
        if int(contact.efc_address) < 0:
            inactive += 1
            continue
        dim = int(contact.dim)
        if dim not in (1, 3, 4, 6):
            raise ValueError("unsupported actual contact dimension")
        coefficient = None
        if dim == 6:
            coefficient = (float(contact.friction[3]), float(contact.friction[4]))
            if any(not np.isfinite(x) or x < 0 for x in coefficient):
                raise ValueError("finite nonnegative actual rolling coefficients required")
        dimensions.append(dim)
        rolling.append(coefficient)
    if not dimensions and inactive:
        status = "INACTIVE_CONTACT_ONLY"
    elif not dimensions:
        status = "NO_OBSERVED_CONTACT"
    elif all(dim < 6 for dim in dimensions):
        status = "NO_ROLLING_CONSTRAINT"
    elif any(x is not None and any(y > 0 for y in x) for x in rolling):
        status = "ROLLING_CONSTRAINT_PRESENT"
    else:
        status = "ZERO_ROLLING_COEFFICIENT"
    return NativeBallGroundContact(status, tuple(dimensions), tuple(rolling), inactive)
