"""Learn new SIM motion directions without overwriting a frozen old subspace."""

from typing import Any

import numpy as np

from rosclaw_soccer.rsi.motion_synergy_reference import fit_motion_synergies


def fit_feedback_complement(
    actual_target_deltas: np.ndarray, backbone_basis: np.ndarray
) -> tuple[np.ndarray, dict[str, Any]]:
    if (
        type(backbone_basis) is not np.ndarray
        or backbone_basis.shape != (6, 29)
        or backbone_basis.dtype != np.float64
        or not np.isfinite(backbone_basis).all()
        or np.linalg.matrix_rank(backbone_basis) != 6
        or type(actual_target_deltas) is not np.ndarray
        or actual_target_deltas.ndim != 2
        or actual_target_deltas.shape[1] != 29
        or actual_target_deltas.dtype != np.float64
        or len(actual_target_deltas) < 60
        or not np.isfinite(actual_target_deltas).all()
    ):
        raise ValueError("complete observed HG29 deltas and finite rank6 frozen backbone required")
    q, _ = np.linalg.qr(backbone_basis.T, mode="reduced")
    residual = actual_target_deltas - (actual_target_deltas @ q) @ q.T
    energy = float(np.square(actual_target_deltas).sum())
    if energy <= 1e-12:
        raise ValueError("nonzero actually executed successful correction data required")
    basis, fit = fit_motion_synergies(residual, rank=6)
    fit["feature_order"] = (
        "actual_paired_successful_PD_target_deltas_outside_frozen_subspace_hg29_rad"
    )
    overlap = float(abs(basis @ q).max())
    if overlap > 1e-8:
        raise ValueError("new directions failed frozen-backbone orthogonality")
    return basis, dict(
        schema="soccer.actual_motor_feedback_complement.v1",
        fit=fit,
        maximum_orthogonal_overlap=overlap,
        outside_backbone_energy_fraction=float(np.square(residual).sum() / energy),
        data_semantics="successful_actual_C_minus_paired_B_PD_target_deltas_before_first_foot_plus10",
        frozen_backbone_modified=False,
        neural_optimizer_updates=0,
        physics_qualified=False,
        hardware_authorized=False,
    )
