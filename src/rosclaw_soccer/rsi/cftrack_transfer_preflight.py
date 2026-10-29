"""Read-only gain and previous-target codec check for CFtrack SIM transfer.

Same-target torque estimates are instantaneous algebra, not an executed
transition, a prediction of balance, or evidence of successful ball control.
"""

import numpy as np

from rosclaw_soccer.rsi.cftrack_action_history import describe_previous_target_encoding


def compare_cftrack_transfer(
    *,
    previous_target: np.ndarray,
    joint_position: np.ndarray,
    joint_velocity: np.ndarray,
    old_stiffness: np.ndarray,
    old_damping: np.ndarray,
    cf_stiffness: np.ndarray,
    cf_damping: np.ndarray,
    default_lab: np.ndarray,
    scale_lab: np.ndarray,
    mj2lab: np.ndarray,
    lab2mj: np.ndarray,
) -> dict:
    values = (
        previous_target,
        joint_position,
        joint_velocity,
        old_stiffness,
        old_damping,
        cf_stiffness,
        cf_damping,
    )
    if any(
        not isinstance(x, np.ndarray)
        or x.shape != (29,)
        or x.dtype.kind != "f"
        or not np.isfinite(x).all()
        for x in values
    ):
        raise ValueError("finite aligned actual HG29 target, state and gains required")
    if any((x <= 0).any() for x in (old_stiffness, old_damping, cf_stiffness, cf_damping)):
        raise ValueError("positive actual and reference gains required")
    encoded = describe_previous_target_encoding(
        previous_target, default_lab, scale_lab, mj2lab, lab2mj
    )
    old = old_stiffness * (previous_target - joint_position) - old_damping * joint_velocity
    new = cf_stiffness * (previous_target - joint_position) - cf_damping * joint_velocity
    delta = np.abs(new - old)
    return dict(
        max_absolute_implied_action=encoded["maximum_absolute_implied_action"],
        target_roundtrip_error_rad=encoded["maximum_target_roundtrip_error_rad"],
        max_same_target_raw_pd_change_nm=float(delta.max()),
        limiting_joint_index=int(delta.argmax()),
        raw_pd_before_effort_clipping=True,
        instantaneous_algebra_not_executed_transition=True,
        input_compatibility_is_not_policy_qualification=True,
        hardware_authorized=False,
        promotion_authorized=False,
    )
