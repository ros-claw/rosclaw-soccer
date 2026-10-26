"""Separate PD intent, measured joint state and applied effort in SIM evidence.

A PD setpoint is not a joint position measurement. This diagnostic neither
changes the strict position-residual contract nor approves a torque interface.
"""

import numpy as np


def audit_pd_intent(
    *,
    decoded: np.ndarray,
    targets: np.ndarray,
    positions: np.ndarray,
    torques: np.ndarray,
    limits: np.ndarray,
    effort: np.ndarray,
    native_steps_per_control: int = 10,
) -> dict:
    if (
        type(native_steps_per_control) is not int
        or native_steps_per_control < 1
        or type(decoded) is not np.ndarray
        or decoded.ndim != 2
        or decoded.shape[1] != 29
        or len(decoded) < 1
    ):
        raise ValueError("nonempty aligned HG29 controller and native measurements required")
    controls = len(decoded)
    shapes = (
        (decoded, (controls, 29)),
        (targets, (controls, 29)),
        (positions, (controls * native_steps_per_control, 29)),
        (torques, (controls * native_steps_per_control, 29)),
        (limits, (29, 2)),
        (effort, (29,)),
    )
    for value, shape in shapes:
        if (
            type(value) is not np.ndarray
            or value.shape != shape
            or value.dtype.kind != "f"
            or not np.isfinite(value).all()
        ):
            raise ValueError("finite aligned floating HG29 arrays required; no truncation")
    if (limits[:, 0] >= limits[:, 1]).any() or (effort <= 0).any():
        raise ValueError("ordered physical limits and positive actual effort caps required")

    def excess(x):
        return np.maximum(np.maximum(limits[:, 0] - x, x - limits[:, 1]), 0.0)

    def record(x):
        e = excess(x)
        return dict(
            rows_outside_range=int((e > 0).any(axis=1).sum()),
            entries_outside_range=int((e > 0).sum()),
            counts_by_hg=(e > 0).sum(axis=0).tolist(),
            maximum_excess_rad_by_hg=e.max(axis=0).tolist(),
            maximum_excess_rad=float(e.max()),
        )

    changed = targets != decoded
    old_out = excess(decoded) > 0
    new_out = excess(targets) > 0
    return dict(
        schema="soccer.sim.pd_intent_audit.v1",
        controls=controls,
        native_steps=len(positions),
        decoded_setpoints=record(decoded),
        submitted_setpoints=record(targets),
        measured_joint_positions=record(positions),
        changed_entries=int(changed.sum()),
        changed_entries_outside_range=int((changed & new_out).sum()),
        changed_entries_already_outside_in_original_nn=int((changed & old_out).sum()),
        maximum_applied_effort_fraction=float((abs(torques) / effort).max()),
        effort_cap_exceeded=bool((abs(torques) > effort + 1e-12).any()),
        intent_is_not_measured_position=True,
        joint_stop_penetration_is_not_setpoint_excess=True,
        safety_or_skill_qualification=False,
        new_neural_calls=0,
        new_physical_steps=0,
        neural_optimizer_updates=0,
        hardware_authorized=False,
    )
