"""Learn full-NN plan->ball response and propose a constrained SIM experiment.

Only a proposal: it MUST pass a new owned NN physics forecast before execution.
No actor weights, runtime permissions, or success labels are produced here.
"""

from typing import Any

import numpy as np
from scipy.optimize import (  # type: ignore[import-untyped]
    Bounds,
    LinearConstraint,
    linprog,
    minimize,
)


def propose_synergy_response(
    basis_hg: np.ndarray,
    samples: list[tuple[np.ndarray, dict[str, Any]]],
) -> tuple[np.ndarray | None, dict[str, Any]]:
    if (
        basis_hg.shape != (6, 29)
        or basis_hg.dtype != np.float64
        or not np.isfinite(basis_hg).all()
        or abs(basis_hg).max() > 1.000001
    ):
        raise ValueError("finite normalized fitted HG29 rank6 basis required")
    rows = []
    for coefficients, window in samples:
        if (
            coefficients.shape != (2, 6)
            or coefficients.dtype != np.float64
            or not np.isfinite(coefficients).all()
            or abs(coefficients).max() > 0.060001
        ):
            raise ValueError("bounded observed two-lobe coefficients required")
        if not window["quality_observed"] or not window["predicted_body_healthy"]:
            continue  # Censored/failed-body trajectories cannot fabricate response labels.
        y = np.array([window["forward_60_m"], window["lateral_60_m"]], dtype=np.float64)
        if not np.isfinite(y).all():
            raise ValueError("finite complete observed ball response required")
        scale = min(1.0, 0.06 / max(float(abs(coefficients @ basis_hg).max()), 1e-20))
        rows.append((coefficients * scale / 0.06, y, window))
    report: dict[str, Any] = dict(
        schema="soccer.synergy_physical_response_model.v1",
        complete_healthy_samples=len(rows),
        neural_updates=0,
        physics_verified=False,
        hardware_authorized=False,
        success_claim=False,
        ridge=0.1,
        target_forward_minimum_m=1.05,
        target_lateral_ratio=0.27,
    )
    clean = [i for i, row in enumerate(rows) if row[2]["predicted_nonfoot_clean"]]
    if len(rows) < 12 or not clean:
        return None, dict(report, status="ABSTAIN_INSUFFICIENT_COMPLETE_HEALTHY_OR_CLEAN_DATA")
    base_id = max(clean, key=lambda i: rows[i][2]["reward"])
    x = np.stack([r[0].ravel() for r in rows])
    y = np.stack([r[1] for r in rows])
    mx, my = x.mean(axis=0), y.mean(axis=0)
    xc, yc = x - mx, y - my
    jac = np.linalg.solve(xc.T @ xc + 0.1 * np.eye(12), xc.T @ yc).T
    base, observed = x[base_id], y[base_id]
    f, lateral = observed
    # Physical reference-trust inequalities, no nonlinear radial-projection surprise.
    joint = np.zeros((58, 12))
    joint[:29, :6], joint[29:, 6:] = basis_hg.T, basis_hg.T
    a = np.concatenate(
        (np.stack((-jac[0], jac[1] - 0.27 * jac[0], -jac[1] - 0.27 * jac[0])), joint, -joint)
    )
    rhs = np.r_[
        f - 1.05, 0.27 * f - lateral, 0.27 * f + lateral, 1 - joint @ base, 1 + joint @ base
    ]
    lo, hi = x.min(axis=0) - base, x.max(axis=0) - base
    report.update(
        base_complete_row=base_id,
        base_effective_coefficients=(base.reshape(2, 6) * 0.06).tolist(),
        base_observed_ball_m=observed.tolist(),
        response_jacobian=jac.tolist(),
        maximum_fit_error_m=float(abs(y - (my + xc @ jac.T)).max()),
        observed_minimum_effective_coefficients=(x.min(axis=0) * 0.06).tolist(),
        observed_maximum_effective_coefficients=(x.max(axis=0) * 0.06).tolist(),
    )
    feasible = linprog(
        np.zeros(12), A_ub=a, b_ub=rhs, bounds=list(zip(lo, hi, strict=True)), method="highs"
    )
    if not feasible.success:
        return None, dict(report, status="ABSTAIN_NO_LINEAR_FEASIBILITY_IN_OBSERVED_TRUST")
    result = minimize(
        lambda d: 0.5 * float(d @ d),
        feasible.x,
        jac=lambda d: d,
        bounds=Bounds(lo, hi),
        constraints=LinearConstraint(a, -np.inf, rhs),
        method="SLSQP",
        options={"ftol": 1e-10, "maxiter": 100},
    )
    if (
        not result.success
        or not np.isfinite(result.x).all()
        or (a @ result.x - rhs > 1e-7).any()
        or (result.x < lo - 1e-7).any()
        or (result.x > hi + 1e-7).any()
    ):
        return None, dict(report, status="ABSTAIN_CONSTRAINED_SOLVE_FAILED")
    proposal = (base + result.x).reshape(2, 6) * 0.06
    if abs(proposal).max() > 0.060001 or abs(proposal @ basis_hg).max() > 0.060001:
        return None, dict(report, status="ABSTAIN_FINAL_TRUST_MISMATCH")
    predicted = observed + jac @ result.x
    return proposal, dict(
        report,
        status="UNVERIFIED_PHYSICAL_EXPERIMENT_PROPOSAL",
        proposed_coefficients=proposal.tolist(),
        predicted_ball_m=predicted.tolist(),
    )
