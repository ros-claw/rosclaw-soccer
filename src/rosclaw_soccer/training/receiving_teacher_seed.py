"""Privileged, seen-trajectory proposals for reachability assay calibration.

This is NOT a causal contact teacher. Future teacher torques may inform the
proposal, and a proposal is not a success certificate. Independently authenticate
the input trajectory, gains, focal-player identity and physical replay before
using the result as evidence. Never use it as a blind-exam policy.
"""

import numpy as np
from numpy.typing import NDArray

from rosclaw_soccer.training.receiving_oracle_schedule import ReceivingOracleSchedule


def teacher_seeded_schedules(
    *,
    agent_id: str,
    entry_frame: int,
    applied_leg_residual: NDArray[np.float64],
    teacher_leg_torque_nm: NDArray[np.float64],
    valid_focal_teacher: NDArray[np.bool_],
    leg_kp: NDArray[np.float64],
) -> tuple[tuple[str, ReceivingOracleSchedule], ...]:
    """Convert a recorded additive PD torque to bounded target-offset seeds.

    ``teacher_leg_torque_nm`` includes the recorded tracking adjustment. The
    boolean mask must identify the focal teacher actually executed in the last
    recorded substep, not merely a teacher that was eligible. This 50 Hz sample
    is not an exact reconstruction of the 500 Hz controller.

    Three proposals retain the original four-knot A0 interface. The fourth has
    27 knots and is a separately labelled temporal-parameterization diagnostic.
    All proposals still pass the original +/-0.1 rad, 0.25-filter, 0.02 rad/step
    execution envelope. The inverse-filter proposal ignores the rate limiter;
    its name must not be interpreted as an exact actuator inverse.
    """
    residual, torque, gains = (
        np.asarray(value) for value in (applied_leg_residual, teacher_leg_torque_nm, leg_kp)
    )
    mask = np.asarray(valid_focal_teacher)
    if any(
        x.dtype.kind not in "fiu" or not np.isfinite(x).all() for x in (residual, torque, gains)
    ):
        raise ValueError("finite numeric residual, torque and gains required")
    if (
        residual.ndim != 2
        or residual.shape[1] != 12
        or not 1 <= len(residual) <= 1000
        or torque.shape != residual.shape
        or mask.shape != (len(residual),)
        or mask.dtype.kind != "b"
        or gains.shape != (12,)
        or np.any(abs(residual) > 0.100000001)
        or np.any(abs(torque) > 300)
        or np.any(gains < 0.001)
        or np.any(gains > 300)
        or type(entry_frame) is not int
        or entry_frame < 0
        or entry_frame + 260 >= len(residual)
    ):
        raise ValueError("bounded 12-joint data and a complete 27-knot source window required")
    # Copy before masking; caller-owned evidence arrays must remain unchanged.
    task_torque = torque.astype(np.float64, copy=True)
    task_torque[~mask] = 0
    desired = residual.astype(np.float64, copy=True) + task_torque / gains
    previous = np.vstack([np.zeros((1, 12)), desired[:-1]])
    inverse = (desired - 0.75 * previous) / 0.25
    lead = np.vstack([desired[5:], np.tile(desired[-1], (5, 1))])
    variants = (
        ("four_sampled", desired, 20, 4),
        ("four_inverse", inverse, 20, 4),
        ("four_lead_100ms", lead, 20, 4),
        ("dense_lead_100ms", lead, 10, 27),
    )
    return tuple(
        (
            name,
            ReceivingOracleSchedule(
                agent_id,
                "A0_leg12",
                entry_frame,
                spacing,
                tuple(
                    tuple(float(v) for v in row)
                    for row in np.clip(
                        target[entry_frame : entry_frame + spacing * count : spacing] / 0.1,
                        -1,
                        1,
                    )
                ),
            ),
        )
        for name, target, spacing, count in variants
    )
