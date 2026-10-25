"""Finite full-NN SIM response identification; proposals are NEVER physical evidence.

Measured predicted outcomes include ball direction AND nonfoot impact severity.
No random search, actor updates, reward relabeling, or real execution authority.
"""

from typing import Any

import numpy as np


def response_features(prediction: dict[str, Any], window: dict[str, Any]) -> np.ndarray | None:
    if not window["predicted_body_healthy"] or not window["quality_observed"]:
        return None
    return np.array(
        [
            window["forward_60_m"],
            window["lateral_60_m"],
            float(
                prediction["ball_forces"][
                    : min(prediction["controls"], 300 - prediction["donor_control_frame"]) * 10,
                    0,
                    2,
                ].max()
            ),
        ],
        dtype=np.float64,
    )


def propose_joint_task_response(
    baseline: np.ndarray | None,
    probes: list[tuple[int, int, np.ndarray | None]],
) -> tuple[np.ndarray | None, dict[str, Any]]:
    report: dict[str, Any] = dict(
        schema="soccer.full_nn_joint_reference_response.v1",
        probe_rad=0.03,
        maximum_l1_proposal_rad=0.06,
        maximum_axis_proposal_rad=0.03,
        forward_target_m=1.05,
        maximum_lateral_ratio=0.27,
        maximum_nonfoot_peak_n=0.5,
        neural_optimizer_updates=0,
        physical_response_verified=False,
        hardware_authorized=False,
        coordinate_box_is_not_jointly_observed_convex_hull=True,
    )
    if len(probes) != 58 or [(i, s) for i, s, _ in probes] != [
        (i, s) for i in range(29) for s in (-1, 1)
    ]:
        raise ValueError("all29 signed full-neural physics probes required")
    for y in (baseline, *(r[2] for r in probes)):
        if y is not None and (
            y.shape != (3,) or y.dtype != np.float64 or not np.isfinite(y).all() or y[2] < 0
        ):
            raise ValueError("finite complete observed ball and nonfoot response required")
    if baseline is None:
        return None, dict(report, status="ABSTAIN_BASELINE_CENSORED_OR_BODY_FAILED")
    axes, jac = [], np.zeros((3, 29))
    curvature = []
    for i in range(29):
        minus, plus = probes[2 * i][2], probes[2 * i + 1][2]
        if minus is None or plus is None:
            continue
        axes.append(i)
        jac[:, i] = (plus - minus) / 0.06
        curvature.append(
            dict(axis=i, midpoint_deviation=(0.5 * (plus + minus) - baseline).tolist())
        )
    report.update(
        complete_axis_pairs=axes,
        baseline_features=baseline.tolist(),
        response_jacobian=jac.tolist(),
        midpoint_curvature=curvature,
    )
    if not axes:
        return None, dict(report, status="ABSTAIN_NO_COMPLETE_FINITE_DIFFERENCE_PAIRS")
    from scipy.optimize import linprog  # type: ignore[import-untyped]

    # x=(29 signed coefficients,29 auxiliary absolute values). A finite L1
    # budget limits coordinated extrapolation; full NN physics must verify it.
    f, lateral, n = baseline
    m = np.stack((-jac[0], jac[1] - 0.27 * jac[0], -jac[1] - 0.27 * jac[0], jac[2], -jac[2]))
    rhs = np.array([f - 1.05, 0.27 * f - lateral, 0.27 * f + lateral, 0.5 - n, n])
    a = np.zeros((5 + 58 + 1, 58))
    a[:5, :29] = m
    a[5:34, :29], a[5:34, 29:] = np.eye(29), -np.eye(29)
    a[34:63, :29], a[34:63, 29:] = -np.eye(29), -np.eye(29)
    a[-1, 29:] = 1
    solved = linprog(
        np.r_[np.zeros(29), np.ones(29)],
        A_ub=a,
        b_ub=np.r_[rhs, np.zeros(58), 0.06],
        bounds=[(-0.03, 0.03) if i in axes else (0, 0) for i in range(29)] + [(0, 0.03)] * 29,
        method="highs",
    )
    if not solved.success or not np.isfinite(solved.x).all():
        return None, dict(report, status="ABSTAIN_NO_BOUNDED_TASK_RESPONSE_FEASIBILITY")
    delta = solved.x[:29]
    if (
        abs(delta).max() > 0.03000001
        or abs(delta).sum() > 0.06000001
        or (a @ solved.x - np.r_[rhs, np.zeros(58), 0.06] > 1e-7).any()
    ):
        raise ValueError("bounded native response solve violated its declared trust")
    coeff = np.zeros((2, 29), dtype=np.float64)
    coeff[0] = delta  # First preparation/swing lobe only; no new recovery authority.
    return coeff, dict(
        report,
        status="UNVERIFIED_FULL_NN_TASK_RESPONSE_PROPOSAL",
        proposed_coefficients=coeff.tolist(),
        predicted_features=(baseline + jac @ delta).tolist(),
    )
