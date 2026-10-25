"""Bounded time conditioning of an owned SIM prior; never physical state edits.

Uses the original position/velocity pairs as cubic Hermite data, including the
chain rule for reference velocity. Body orientations use shortest-arc SLERP.
The zero branch does not resample or replace any array (bit-exact legacy path).
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation, Slerp  # type: ignore[import-untyped]


@dataclass(frozen=True)
class MotionTimingPlan:
    start_frame: int
    intervals: int
    peak_shift_frames: float

    def validate(self) -> None:
        if (
            type(self.start_frame) is not int
            or self.start_frame < 0
            or type(self.intervals) is not int
            or not 24 <= self.intervals <= 64
            or type(self.peak_shift_frames) not in (int, float)
            or not np.isfinite(self.peak_shift_frames)
            or abs(self.peak_shift_frames) > 3
        ):
            raise ValueError("bounded data-only motion timing plan required")


def condition_motion_reference(policy: Any, plan: MotionTimingPlan) -> dict[str, Any]:
    """Replace reference arrays ONLY, before this boundary's policy proposal.

    Caller must own the pure SIM policy and guarantee the pre-proposal boundary.
    The first/last sample and everything outside the plan remain original bits.
    No future state/history is adopted and no pretrained network is modified.
    """
    if type(plan) is not MotionTimingPlan:
        raise ValueError("explicit motion timing plan required")
    plan.validate()
    now = int(policy.time_step) - int(policy.WARMUP_STEPS)
    end = plan.start_frame + plan.intervals
    if (
        policy.runtime_mode != "sim"
        or policy.use_body_frame_ball
        or policy.control_dt != 0.02
        or now < 0
        or plan.start_frame < now
        or end >= policy.motion_total_steps
    ):
        raise ValueError("future complete original50Hz SIM reference window required")
    receipt = dict(
        schema="soccer.motion_reference_timing.v1",
        **vars(plan),
        physical_state_modified=False,
        physical_clock_modified=False,
        network_updates=0,
        hardware_authorized=False,
    )
    if plan.peak_shift_frames == 0:
        return dict(receipt, applied=False, zero_branch_exact=True)
    names = ("motion_joint_pos", "motion_joint_vel", "motion_body_pos", "motion_body_quat")
    arrays = [getattr(policy, name) for name in names]
    n = policy.motion_total_steps
    q, v, body, quat = arrays
    if (
        any(type(a) is not np.ndarray or a.dtype.kind != "f" for a in arrays)
        or q.shape != (n, 29)
        or v.shape != q.shape
        or body.ndim != 3
        or body.shape[0] != n
        or body.shape[2] != 3
        or quat.shape != (*body.shape[:2], 4)
        or any(not np.isfinite(a).all() for a in arrays)
        or np.max(abs(np.linalg.norm(quat, axis=2) - 1)) > 1e-4
    ):
        raise ValueError("finite original joint pairs/body poses and unit wxyz required")
    rows = np.arange(plan.start_frame + 1, end)
    phase = (rows - plan.start_frame) / plan.intervals
    sample = rows + plan.peak_shift_frames * np.sin(np.pi * phase) ** 2
    rate = 1 + plan.peak_shift_frames * np.pi / plan.intervals * np.sin(2 * np.pi * phase)
    if (rate <= 0).any() or sample.min() < plan.start_frame or sample.max() > end:
        raise ValueError("strictly forward in-window reference clock required")
    lo = np.floor(sample).astype(int)
    hi = lo + 1
    u = (sample - lo)[:, None]
    dt = policy.control_dt
    a, b = q[lo].astype(float), q[hi].astype(float)
    va, vb = v[lo].astype(float) * dt, v[hi].astype(float) * dt
    pos = (2 * u**3 - 3 * u**2 + 1) * a + (u**3 - 2 * u**2 + u) * va
    pos += (-2 * u**3 + 3 * u**2) * b + (u**3 - u**2) * vb
    vel = (6 * u**2 - 6 * u) * a + (3 * u**2 - 4 * u + 1) * va
    vel += (-6 * u**2 + 6 * u) * b + (3 * u**2 - 2 * u) * vb
    copies = [a.copy() for a in arrays]
    copies[0][rows] = pos
    copies[1][rows] = vel * rate[:, None] / dt
    weight = u[:, :, None]
    copies[2][rows] = body[lo] + weight * (body[hi] - body[lo])
    # Reuse the same scipy shortest-arc wxyz/xyzw conversion as SONIC preparation.
    timeline = np.arange(n, dtype=float)
    for j in range(body.shape[1]):
        rotations = Rotation.from_quat(quat[:, j, [1, 2, 3, 0]])
        copies[3][rows, j] = Slerp(timeline, rotations)(sample).as_quat()[:, [3, 0, 1, 2]]
    if any(not np.isfinite(a).all() for a in copies):
        raise ValueError("nonfinite motion conditioning rejected before installation")
    for name, array in zip(names, copies, strict=True):
        setattr(policy, name, array)
    return dict(
        receipt, applied=True, minimum_rate=float(rate.min()), maximum_rate=float(rate.max())
    )
