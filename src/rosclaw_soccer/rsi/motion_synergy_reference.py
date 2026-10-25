"""Data-derived multiframe SIM reference coordination, not actuator commands.

Successful executed HG29 target innovations define a low-rank motor subspace.
Two smooth temporal lobes can coordinate preparation and swing/recovery. The
original NN still closes the measured-body loop and owns the actual PD target.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np


def fit_motion_synergies(
    innovations: np.ndarray, *, rank: int = 6
) -> tuple[np.ndarray, dict[str, Any]]:
    """PCA of observed target innovations, never fabricated successful labels."""
    if (
        type(innovations) is not np.ndarray
        or innovations.ndim != 2
        or innovations.shape[1] != 29
        or len(innovations) < 60
        or innovations.dtype != np.float64
        or not np.isfinite(innovations).all()
        or type(rank) is not int
        or not 1 <= rank <= 12
    ):
        raise ValueError("finite observed HG29 innovations and bounded rank required")
    centered = innovations - innovations.mean(axis=0)
    _, singular, vt = np.linalg.svd(centered, full_matrices=False)
    if singular[rank - 1] < 1e-8:
        raise ValueError("observed motion data has insufficient nondegenerate rank")
    basis = vt[:rank].copy()
    for row in basis:
        # Canonical signs avoid making the CEM direction depend on SVD sign ambiguity.
        row *= 1 if row[np.argmax(abs(row))] > 0 else -1
        row /= abs(row).max()
    return basis, dict(
        schema="soccer.executed_motion_synergies.v1",
        samples=len(innovations),
        rank=rank,
        singular_values=singular.tolist(),
        explained_variance_fraction=float((singular[:rank] ** 2).sum() / (singular**2).sum()),
        feature_order="canonical_hg29_target_innovation_rad_per_control",
        actor_updates=0,
        physics_qualified=False,
        hardware_authorized=False,
    )


@dataclass(frozen=True)
class MotionSynergyPlan:
    start_frame: int
    # Basis is already converted to the ORIGINAL prior's Isaac joint order.
    basis_isaac: tuple[tuple[float, ...], ...]
    coefficients: tuple[tuple[float, ...], tuple[float, ...]]

    def offsets(self) -> tuple[np.ndarray, np.ndarray]:
        if type(self.start_frame) is not int or self.start_frame < 0:
            raise ValueError("future integer reference frame required")
        if (
            type(self.basis_isaac) is not tuple
            or type(self.coefficients) is not tuple
            or any(type(r) is not tuple for r in (*self.basis_isaac, *self.coefficients))
        ):
            raise ValueError("immutable data-only synergy plan required")
        basis, coeff = np.array(self.basis_isaac), np.array(self.coefficients)
        if (
            basis.ndim != 2
            or basis.shape[1] != 29
            or not 1 <= len(basis) <= 12
            or coeff.shape != (2, len(basis))
            or basis.dtype.kind != "f"
            or coeff.dtype.kind != "f"
            or not np.isfinite(basis).all()
            or not np.isfinite(coeff).all()
            or abs(basis).max() > 1.000001
            or abs(coeff).max() > 0.060001
        ):
            raise ValueError("finite normalized basis and bounded coefficients required")
        # Two24-interval C1 lobes, entry/transition/exit EXACTLY zero and zero velocity.
        pos, vel = np.zeros((49, 29)), np.zeros((49, 29))
        u = np.arange(1, 24) / 24
        for lobe in range(2):
            direction = coeff[lobe] @ basis
            rows = lobe * 24 + np.arange(1, 24)
            pos[rows] = np.sin(np.pi * u)[:, None] ** 2 * direction
            vel[rows] = (np.pi / 24 / 0.02 * np.sin(2 * np.pi * u))[:, None] * direction
        maximum = float(abs(pos).max())
        # Declared whole-plan radial trust projection, no per-joint clipping or repairs.
        scale = min(1.0, 0.06 / max(maximum, 1e-20))
        return pos * scale, vel * scale


def condition_synergy_reference(policy: Any, plan: MotionSynergyPlan) -> dict[str, Any]:
    if type(plan) is not MotionSynergyPlan:
        raise ValueError("explicit immutable motion synergy plan required")
    pos, vel = plan.offsets()
    now = int(policy.time_step) - int(policy.WARMUP_STEPS)
    if (
        policy.runtime_mode != "sim"
        or policy.use_body_frame_ball
        or policy.control_dt != 0.02
        or now < 0
        or plan.start_frame < now
        or plan.start_frame + 48 >= policy.motion_total_steps
    ):
        raise ValueError("complete future original50Hz SIM reference required")
    receipt = dict(
        schema="soccer.motion_synergy_reference.v1",
        start_frame=plan.start_frame,
        intervals=48,
        coefficients=plan.coefficients,
        physical_state_modified=False,
        network_updates=0,
        hardware_authorized=False,
    )
    if not np.any(pos):
        return dict(receipt, applied=False, zero_branch_exact=True)
    q, v = policy.motion_joint_pos, policy.motion_joint_vel
    if (
        q.shape != (policy.motion_total_steps, 29)
        or v.shape != q.shape
        or q.dtype.kind != "f"
        or v.dtype.kind != "f"
        or not np.isfinite(q).all()
        or not np.isfinite(v).all()
    ):
        raise ValueError("finite original reference position/velocity required")
    new_q, new_v = q.copy(), v.copy()
    rows = plan.start_frame + np.r_[1:24, 25:48]
    local = rows - plan.start_frame
    new_q[rows] += pos[local].astype(q.dtype)
    new_v[rows] += vel[local].astype(v.dtype)
    if not np.isfinite(new_q).all() or not np.isfinite(new_v).all():
        raise ValueError("nonfinite conditioned reference rejected")
    policy.motion_joint_pos, policy.motion_joint_vel = new_q, new_v
    return dict(receipt, applied=True, maximum_reference_offset_rad=float(abs(pos).max()))
