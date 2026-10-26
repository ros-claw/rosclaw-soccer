"""Explicit absolute IK waypoint reference contract, distinct from .35rad deltas.

The coordinate is an absolute physical leg configuration, not a correction to
another body pose. Physical reference joint ranges and a NEW declared6rad/s
additional reference velocity envelope apply. No actual motor/PD limit changes.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np

from rosclaw_soccer.rsi.contact_reference_plan import _c2_offsets


@dataclass(frozen=True)
class AbsoluteContactReferencePlan:
    start_frame: int
    hg_to_isaac: tuple[int, ...]
    waypoint_hg: tuple[float, ...]
    peak_control: int
    bounded_curve: bool = False

    def offsets(
        self, policy: Any, *, limits_hg: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        if (
            type(self.start_frame) is not int
            or self.start_frame < 0
            or type(self.peak_control) is not int
            or not 1 <= self.peak_control < 48
            or type(self.hg_to_isaac) is not tuple
            or len(self.hg_to_isaac) != 29
            or any(type(i) is not int for i in self.hg_to_isaac)
            or sorted(self.hg_to_isaac) != list(range(29))
            or type(self.waypoint_hg) is not tuple
            or len(self.waypoint_hg) != 29
            or any(
                type(v) is not float or not np.isfinite(v) or abs(v) > 10 for v in self.waypoint_hg
            )
            or type(policy.motion_joint_pos) is not np.ndarray
            or policy.motion_joint_pos.shape != (policy.motion_total_steps, 29)
            or self.start_frame + 48 >= policy.motion_total_steps
            or type(self.bounded_curve) is not bool
        ):
            raise ValueError("explicit complete absolute-contact reference contract required")
        nominal = policy.motion_joint_pos[
            self.start_frame + self.peak_control, self.hg_to_isaac
        ].astype(np.float64)
        delta = np.zeros(29)
        delta[:12] = np.array(self.waypoint_hg)[:12] - nominal[:12]
        if self.bounded_curve:
            from rosclaw_soccer.rsi.bounded_contact_reference_curve import bounded_contact_curve

            if limits_hg is None:
                raise ValueError("bounded absolute curve requires actual physical joint ranges")
            q = policy.motion_joint_pos[
                self.start_frame : self.start_frame + 49, self.hg_to_isaac
            ].astype(np.float64)
            pos_hg, vel_hg, _ = bounded_contact_curve(
                q, np.array(self.waypoint_hg), limits_hg, self.peak_control
            )
            pos, vel = np.zeros((49, 29)), np.zeros((49, 29))
            pos[:, self.hg_to_isaac], vel[:, self.hg_to_isaac] = pos_hg, vel_hg
            return pos, vel
        if (
            not np.isfinite(delta).all()
            or 1.875 * abs(delta).max() / min(self.peak_control, 48 - self.peak_control) / 0.02
            > 6.0 + 1e-12
        ):
            raise ValueError("absolute waypoint exceeds NEW declared6rad/s reference velocity cap")
        return _c2_offsets(delta, self.hg_to_isaac, self.peak_control)


def condition_absolute_contact_reference(
    policy: Any, plan: AbsoluteContactReferencePlan, *, limits_hg: np.ndarray
) -> dict[str, Any]:
    if type(plan) is not AbsoluteContactReferencePlan:
        raise ValueError("explicit absolute SIM reference required")
    pos, vel = plan.offsets(policy, limits_hg=limits_hg)
    q, v = policy.motion_joint_pos, policy.motion_joint_vel
    if (
        policy.runtime_mode != "sim"
        or policy.use_body_frame_ball
        or policy.control_dt != 0.02
        or int(policy.time_step) - int(policy.WARMUP_STEPS) != plan.start_frame
        or any(
            type(x) is not np.ndarray
            or x.shape != (policy.motion_total_steps, 29)
            or x.dtype.kind != "f"
            or not np.isfinite(x).all()
            for x in (q, v)
        )
        or type(limits_hg) is not np.ndarray
        or limits_hg.shape != (29, 2)
        or limits_hg.dtype != np.float64
        or not np.isfinite(limits_hg).all()
        or (limits_hg[:, 0] >= limits_hg[:, 1]).any()
    ):
        raise ValueError("current SIM49 window and physical HG reference ranges required")
    receipt = dict(
        schema="soccer.absolute_contact_reference.v1",
        start_frame=plan.start_frame,
        peak_control=plan.peak_control,
        waypoint_hg=plan.waypoint_hg,
        maximum_reference_offset_rad=float(abs(pos).max()),
        maximum_additional_reference_velocity_rad_s=float(abs(vel).max()),
        additional_reference_velocity_limit_rad_s=6.0,
        reference_authority="absolute_physical_leg_configuration_not_035_delta",
        entry_reference_unchanged=True,
        past_reference_unchanged=True,
        original_anchor_unchanged=True,
        physical_state_modified=False,
        executed_targets_overridden=False,
        neural_weights_modified=False,
        hardware_authorized=False,
    )
    if plan.bounded_curve:
        receipt["trajectory_contract"] = "bounded_c2_bspline_v1"
        receipt["continuous_additional_velocity_bound_rad_s"] = 6.0
        receipt["physical_joint_bounds_scope"] = "49_consumed_reference_samples"
        receipt["trajectory_linear_programs_per_resolution"] = 12
    if not np.any(pos):
        return dict(receipt, applied=False, zero_branch_exact=True)
    start = plan.start_frame
    new_q, new_v = q.copy(), v.copy()
    new_q[start : start + 49] += pos.astype(q.dtype)
    new_v[start : start + 49] += vel.astype(v.dtype)
    changed = pos[:, plan.hg_to_isaac] != 0
    actual = new_q[start : start + 49, plan.hg_to_isaac]
    if ((actual < limits_hg[:, 0]) & changed).any() or ((actual > limits_hg[:, 1]) & changed).any():
        return dict(
            receipt, applied=False, abstention="ABSOLUTE_REFERENCE_OUTSIDE_PHYSICAL_JOINT_LIMIT"
        )
    if not np.isfinite(new_q).all() or not np.isfinite(new_v).all():
        raise ValueError("nonfinite absolute reference, no partial installation")
    endpoint_error = float(
        abs(actual[plan.peak_control, :12] - np.array(plan.waypoint_hg)[:12]).max()
    )
    if endpoint_error > 1e-6:
        raise ValueError("encoded absolute reference failed waypoint commitment")
    policy.motion_joint_pos, policy.motion_joint_vel = new_q, new_v
    return dict(receipt, applied=True, encoded_waypoint_error_rad=endpoint_error)
