"""Bounded experiments derived from actual solver-contact radial heading.

An approximate geometric inverse, not a policy or a predicted save/goal.
The original IK joint cap and body controller are unchanged. Native execution
must test the coupling from requested foot rotation to actual contact normal.
"""

import math

import numpy as np


def contact_yaw_proposals(
    base: np.ndarray, *, center: int, contact_control: int, radial_heading_degrees: float
) -> tuple[np.ndarray, dict[str, object]]:
    if (
        base.shape != (2, 6)
        or not np.isfinite(base).all()
        or (np.abs(base) > 1).any()
        or type(center) is not int
        or not 30 <= center <= 200
        or type(contact_control) is not int
        or not 0 <= contact_control < 300
        or not math.isfinite(radial_heading_degrees)
        or abs(radial_heading_degrees) > 60
    ):
        raise ValueError("actual bounded TRAIN foot intent and contact heading required")
    t = (contact_control - np.array([center - 12, center + 8])) / 30.0
    weights = np.where(np.abs(t) < 1, (1 - t * t) ** 3, 0.0)
    if weights.sum() < 0.25:
        raise ValueError("contact outside source foot-intent authority window")
    delta = -math.radians(radial_heading_degrees) / (0.15 * weights.sum())
    multipliers = np.array([0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0, 1.125, -0.25])
    proposed = np.repeat(base[None], len(multipliers), axis=0)
    proposed[:, :, 5] = np.clip(base[None, :, 5] + multipliers[:, None] * delta, -1, 1)
    return proposed, dict(
        actual_contact_radial_heading_degrees=radial_heading_degrees,
        actual_contact_control=contact_control,
        center=center,
        phase_weights=weights.tolist(),
        yaw_pose_scale_rad=0.15,
        inverse_delta_per_basis=delta,
        multipliers=multipliers.tolist(),
        non_yaw_latents_unchanged=True,
        exact_inverse_dynamics_claim=False,
        physical_success_predicted=False,
    )
