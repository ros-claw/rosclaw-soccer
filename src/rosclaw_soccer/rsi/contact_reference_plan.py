"""C2 contact-conditioned joint reference, never a physical target playback.

Reference authority is explicitly distinct from executed-target residuals.
The original NN still reads its own body and produces every actual HG target.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np


def _c2_offsets(
    delta: np.ndarray, mapping: tuple[int, ...], peak: int
) -> tuple[np.ndarray, np.ndarray]:
    """Shared curve math only; authority validation belongs to each contract."""
    indices = np.arange(49, dtype=np.float64)
    up = indices <= peak
    u = np.where(up, indices / peak, (indices - peak) / (48 - peak))
    s = 10 * u**3 - 15 * u**4 + 6 * u**5
    derivative = 30 * u**2 - 60 * u**3 + 30 * u**4
    fraction = np.where(up, s, 1 - s)
    rate = np.where(up, derivative / (peak * 0.02), -derivative / ((48 - peak) * 0.02))
    pos, vel = np.zeros((49, 29)), np.zeros((49, 29))
    pos[:, mapping] = fraction[:, None] * delta
    vel[:, mapping] = rate[:, None] * delta
    return pos, vel


@dataclass(frozen=True)
class ContactReferencePlan:
    start_frame: int
    hg_to_isaac: tuple[int, ...]
    delta_hg: tuple[float, ...]
    peak_control: int

    def offsets(self) -> tuple[np.ndarray, np.ndarray]:
        if (
            type(self.start_frame) is not int
            or self.start_frame < 0
            or type(self.peak_control) is not int
            or not 1 <= self.peak_control < 48
            or type(self.hg_to_isaac) is not tuple
            or len(self.hg_to_isaac) != 29
            or any(type(i) is not int for i in self.hg_to_isaac)
            or sorted(self.hg_to_isaac) != list(range(29))
            or type(self.delta_hg) is not tuple
            or len(self.delta_hg) != 29
            or any(type(v) is not float or not np.isfinite(v) for v in self.delta_hg)
        ):
            raise ValueError("immutable bounded contact reference and HG29 permutation required")
        delta = np.array(self.delta_hg, np.float64)
        if abs(delta).max() > 0.350001 or np.any(delta[12:]):
            raise ValueError("explicit leg-only .35rad reference authority required")
        # Include the analytic derivative maximum, not only sampled ticks.
        if (
            1.875 * abs(delta).max() / min(self.peak_control, 48 - self.peak_control) / 0.02
            > 4.0 + 1e-12
        ):
            raise ValueError("contact reference exceeds declared additional4rad/s velocity cap")
        return _c2_offsets(delta, self.hg_to_isaac, self.peak_control)


def condition_contact_reference(
    policy: Any, plan: ContactReferencePlan, *, limits_hg: np.ndarray
) -> dict[str, Any]:
    if type(plan) is not ContactReferencePlan:
        raise ValueError("explicit immutable SIM contact reference required")
    pos, vel = plan.offsets()
    now = int(policy.time_step) - int(policy.WARMUP_STEPS)
    q, v = policy.motion_joint_pos, policy.motion_joint_vel
    if (
        policy.runtime_mode != "sim"
        or policy.use_body_frame_ball
        or policy.control_dt != 0.02
        or now != plan.start_frame
        or plan.start_frame + 48 >= policy.motion_total_steps
        or any(
            type(a) is not np.ndarray
            or a.shape != (policy.motion_total_steps, 29)
            or a.dtype.kind != "f"
            or not np.isfinite(a).all()
            for a in (q, v)
        )
        or type(limits_hg) is not np.ndarray
        or limits_hg.shape != (29, 2)
        or limits_hg.dtype != np.float64
        or not np.isfinite(limits_hg).all()
        or (limits_hg[:, 0] >= limits_hg[:, 1]).any()
    ):
        raise ValueError(
            "current50Hz SIM reference, complete49window and physical HG limits required"
        )
    receipt = dict(
        schema="soccer.contact_reference_plan.v1",
        start_frame=now,
        peak_control=plan.peak_control,
        delta_hg=plan.delta_hg,
        maximum_reference_offset_rad=float(abs(pos).max()),
        maximum_additional_reference_velocity_rad_s=float(abs(vel).max()),
        reference_offset_limit_rad=0.35,
        reference_velocity_limit_rad_s=4.0,
        entry_reference_unchanged=True,
        past_reference_unchanged=True,
        original_anchor_unchanged=True,
        physical_state_modified=False,
        executed_targets_overridden=False,
        neural_weights_modified=False,
        hardware_authorized=False,
    )
    if not np.any(pos):
        return dict(receipt, applied=False, zero_branch_exact=True)
    start = plan.start_frame
    new_q, new_v = q.copy(), v.copy()
    new_q[start : start + 49] += pos.astype(q.dtype)
    new_v[start : start + 49] += vel.astype(v.dtype)
    changed = pos[:, plan.hg_to_isaac] != 0
    actual = new_q[start : start + 49, plan.hg_to_isaac]
    if ((actual < limits_hg[:, 0]) & changed).any() or ((actual > limits_hg[:, 1]) & changed).any():
        return dict(receipt, applied=False, abstention="REFERENCE_OUTSIDE_PHYSICAL_JOINT_LIMIT")
    if not np.isfinite(new_q).all() or not np.isfinite(new_v).all():
        raise ValueError("nonfinite conditioned reference, no partial installation")
    policy.motion_joint_pos, policy.motion_joint_vel = new_q, new_v
    return dict(receipt, applied=True)
