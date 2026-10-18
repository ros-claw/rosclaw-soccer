"""Fixed-base local compliance shaping, NOT floating-base stability proof.

The rank-one contraction preserves the selected leg's linearized Cartesian
X/Y compliance and increases only its Z compliance. All damping and other
joints are unchanged. Dynamic G1 ball outcomes require actual SIM experiments.
"""

from typing import Any

import numpy as np

from rosclaw_soccer.providers.g1.joint_contract import G1_DDS_JOINT_NAMES
from rosclaw_soccer.rsi.taskspace_swing_evidence import LEG_NAMES


def vertical_compliance_stiffness(
    kp: np.ndarray, material_point_jacobian: np.ndarray, side: int | None
) -> tuple[np.ndarray, dict[str, Any]]:
    if (
        not isinstance(kp, np.ndarray)
        or kp.shape != (29,)
        or kp.dtype != np.float64
        or not np.isfinite(kp).all()
        or np.any(kp < 1e-6)
        or np.any(kp > 2000)
        or not isinstance(material_point_jacobian, np.ndarray)
        or material_point_jacobian.shape != (3, 29)
        or material_point_jacobian.dtype != np.float64
        or not np.isfinite(material_point_jacobian).all()
        or np.max(abs(material_point_jacobian)) > 10
        or (side is not None and (type(side) is not int or side not in (0, 1)))
    ):
        raise ValueError(
            "finite canonical gains, current material Jacobian and selected leg required"
        )
    matrix = np.diag(kp)
    report: dict[str, Any] = dict(
        schema="soccer.rsi.vertical_compliance_stiffness.v1",
        side=side,
        contraction_fraction=0.5,
        enabled=False,
        relative_stiffness_eigenvalue_bounds=[0.5, 1.0],
        interpretation="fixed-base selected-leg local Cartesian compliance, not dynamic stability",
        damping_changed=False,
        hardware_authorized=False,
        promotion_authorized=False,
    )
    if side is None:
        report["status"] = "NO_PRESSURE_NO_CHANGE"
        return matrix, report
    ids = [G1_DDS_JOINT_NAMES.index(name) for name in LEG_NAMES[side]]
    jac = material_point_jacobian[:, ids]
    scale = np.sqrt(kp[ids])
    # Force directions in stiffness-whitened joint coordinates: S^-1 J^T.
    directions = jac.T / scale[:, None]
    u, singular, _ = np.linalg.svd(directions[:, :2], full_matrices=False)
    rank = int(np.count_nonzero(singular > max(1e-12, float(singular.max()) * 1e-10)))
    tangent = u[:, :rank]
    normal = directions[:, 2] - tangent @ (tangent.T @ directions[:, 2])
    norm = float(np.linalg.norm(normal))
    report.update(tangential_rank=rank, independent_vertical_direction_norm=norm)
    if norm <= 1e-10:
        report["status"] = "DEGENERATE_GEOMETRY_NO_CHANGE"
        return matrix, report
    normal /= norm
    weighted = scale * normal
    block = np.diag(kp[ids]) - 0.5 * np.outer(weighted, weighted)
    matrix[np.ix_(ids, ids)] = block
    parent_compliance = jac @ np.diag(1.0 / kp[ids]) @ jac.T
    candidate_compliance = jac @ np.linalg.solve(block, jac.T)
    relative = block / scale[:, None] / scale[None, :]
    error = float(np.max(abs((candidate_compliance - parent_compliance)[:2])))
    if (
        np.linalg.eigvalsh(relative).min() < 0.5 - 1e-10
        or np.linalg.eigvalsh(relative).max() > 1.0 + 1e-10
        or error > 1e-10
        or candidate_compliance[2, 2] < parent_compliance[2, 2] - 1e-12
    ):
        raise ValueError("bounded stiffness contraction and tangential compliance identity failed")
    report.update(
        status="VERTICAL_COMPLIANCE_ONLY",
        enabled=True,
        changed_joint_indices=ids,
        maximum_cartesian_tangential_compliance_change_m_per_n=error,
        parent_cartesian_compliance_m_per_n=parent_compliance.tolist(),
        candidate_cartesian_compliance_m_per_n=candidate_compliance.tolist(),
        relative_stiffness_eigenvalues=np.linalg.eigvalsh(relative).tolist(),
    )
    return matrix, report
