"""SIM-only causal candidate for joint foot interception and shin clearance.

This is a bounded unlearned teacher proposal, not a physics prediction, a
trained policy, or a certificate of successful or safe contact.
"""

import numpy as np


def predictive_foot_shin_target(
    *,
    nominal: np.ndarray,
    previous_delta: np.ndarray,
    joint_ranges: np.ndarray,
    foot_jacobian: np.ndarray,
    foot_displacement: np.ndarray,
    shin_gap: np.ndarray,
    shin_gap_rate: np.ndarray,
    shin_gap_jacobian: np.ndarray,
    foot: int,
    enabled: bool,
    horizon_sec: float = 0.12,
    target_gap_m: float = 0.05,
    max_delta_rad: float = 0.16,
    slew_rad: float = 0.025,
    axis_local_fallback: bool = False,
    parent_relative_envelope: bool = False,
    freeze_ankle_correction: bool = False,
) -> dict:
    """Compute one damped, bounded leg-only correction from current measurements.

    Gap rate is the signed surface separation rate, not a future observation.
    A negative rate is closing. The linear extrapolation can be wrong after
    impact, so only real simulation can judge the resulting trajectory.
    """
    arrays = (
        (nominal, (29,)),
        (previous_delta, (29,)),
        (joint_ranges, (29, 2)),
        (foot_jacobian, (3, 29)),
        (foot_displacement, (3,)),
        (shin_gap, (2,)),
        (shin_gap_rate, (2,)),
        (shin_gap_jacobian, (2, 29)),
    )
    if (
        any(
            not isinstance(x, np.ndarray)
            or x.shape != shape
            or x.dtype != np.float64
            or not np.isfinite(x).all()
            for x, shape in arrays
        )
        or type(foot) is not int
        or foot not in (0, 1)
        or type(enabled) is not bool
        or type(axis_local_fallback) is not bool
        or type(parent_relative_envelope) is not bool
        or type(freeze_ankle_correction) is not bool
        or any(
            type(v) is not float or not np.isfinite(v)
            for v in (horizon_sec, target_gap_m, max_delta_rad, slew_rad)
        )
        or not 0.02 <= horizon_sec <= 0.2
        or not 0.01 <= target_gap_m <= 0.1
        or not 0 < max_delta_rad <= 0.2
        or not 0 < slew_rad <= 0.03
        or np.any(joint_ranges[:, 0] >= joint_ranges[:, 1])
        or np.max(np.abs(previous_delta)) > max_delta_rad + 1e-12
        or np.any(previous_delta[np.r_[12:29]] != 0)
        or np.any(previous_delta[6 * (1 - foot) : 6 * (1 - foot) + 6] != 0)
        or (freeze_ankle_correction and np.any(previous_delta[[4, 5, 10, 11]] != 0))
        or np.linalg.norm(foot_displacement) > 0.08000001
    ):
        raise ValueError("finite aligned bounded predictive contact inputs required")
    leg = slice(6 * foot, 6 * foot + 6)
    # If the parent is already outside an actuator range, an experimental
    # correction may move it toward that range, but never farther away.
    # This does not assert that the resulting absolute target is legal.
    effective_low = np.minimum(joint_ranges[leg, 0], nominal[leg])
    effective_high = np.maximum(joint_ranges[leg, 1], nominal[leg])
    range_low = effective_low if parent_relative_envelope else joint_ranges[leg, 0]
    range_high = effective_high if parent_relative_envelope else joint_ranges[leg, 1]
    predicted_gap = shin_gap + horizon_sec * shin_gap_rate
    desired_gap = np.clip(target_gap_m - predicted_gap, 0.0, 0.08)
    lower = np.maximum.reduce(
        (
            np.full(6, -max_delta_rad),
            previous_delta[leg] - slew_rad,
            range_low - nominal[leg],
        )
    )
    upper = np.minimum.reduce(
        (
            np.full(6, max_delta_rad),
            previous_delta[leg] + slew_rad,
            range_high - nominal[leg],
        )
    )
    invalid = lower > upper
    outside_parent = (nominal[leg] < joint_ranges[leg, 0]) | (nominal[leg] > joint_ranges[leg, 1])
    # A parent target can already lie outside the physical range. An axis
    # that has never received our correction is frozen exactly at that parent,
    # not repaired or falsely certified. If our own previous nonzero correction
    # cannot meet slew and physical limits simultaneously, that axis returns to
    # parent and records the unavoidable jump. Other feasible axes retain their
    # own constraints; do not abruptly erase the entire leg's correction.
    fallback = bool(np.any(invalid & (previous_delta[leg] != 0)))
    frozen = invalid | (
        outside_parent & (previous_delta[leg] == 0) & (not parent_relative_envelope)
    )
    if freeze_ankle_correction:
        frozen[4:6] = True
    free = ~frozen
    requested = np.zeros(29, dtype=np.float64)
    if enabled and np.any(free) and (axis_local_fallback or not fallback):
        # Only feasible axes participate in the task solution.
        jac = np.vstack((foot_jacobian[:, leg], 2.0 * shin_gap_jacobian[:, leg]))[:, free]
        desired = np.concatenate((foot_displacement, 2.0 * desired_gap))
        leg_request = requested[leg]
        leg_request[free] = np.linalg.solve(
            jac.T @ jac + 0.04**2 * np.eye(int(free.sum())), jac.T @ desired
        )
    delta = np.zeros(29, dtype=np.float64)
    leg_delta = delta[leg]
    if axis_local_fallback or not fallback:
        leg_delta[free] = np.clip(requested[leg][free], lower[free], upper[free])
    target = nominal.copy() if not np.any(delta) else nominal + delta
    applied = target - nominal
    delta_jump = float(np.max(np.abs(applied - previous_delta)))
    return dict(
        target=target,
        applied_delta=applied,
        requested_delta=requested,
        predicted_shin_gap=predicted_gap,
        desired_shin_clearance=desired_gap,
        fallback=fallback,
        frozen_parent_axes=frozen,
        delta_jump_rad=delta_jump,
        slew_limit_respected=bool(delta_jump <= slew_rad + 1e-12),
        enabled=enabled,
    )
