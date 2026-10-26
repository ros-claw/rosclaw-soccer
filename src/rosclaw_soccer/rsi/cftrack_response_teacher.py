"""A fitted local response model and one bounded SIM planning proposal.

This is not neural weight training. Unsafe calibration directions are excluded;
unmeasured joint coupling/extrapolation always needs a new native NN preview.
"""

import re
from dataclasses import dataclass
from typing import Any

import numpy as np

from rosclaw_soccer.physics.small_strict_qp import solve_small_strict_qp
from rosclaw_soccer.rsi.cftrack_motor_authority import MOTOR_AXES
from rosclaw_soccer.rsi.cftrack_motor_residual_plan import CFtrackMotorResidualPlan
from rosclaw_soccer.rsi.contact_reference_plan import _c2_offsets


@dataclass(frozen=True)
class MotorResponseModel:
    jacobian: tuple[tuple[float, ...], ...]
    lower: tuple[float, ...]
    upper: tuple[float, ...]
    source_pilot_hash: str
    fitted_arms: tuple[tuple[str, ...], ...]


def fit_motor_response(pilot: dict[str, Any], *, source_hash: str) -> MotorResponseModel:
    if (
        type(source_hash) is not str
        or re.fullmatch(r"sha256:[0-9a-f]{64}", source_hash) is None
        or pilot.get("status") != "CFTRACK_MOTOR_AUTHORITY_LOCAL_RESCUE"
        or len(pilot.get("records", [])) != 9
    ):
        raise ValueError("complete bound nine-arm calibration required")
    rows = pilot["records"]
    if [r["name"] for r in rows] != ["parent", *[f"arm-{i:02d}" for i in range(8)]]:
        raise ValueError("declared calibration identities required")

    def output(row: dict[str, Any]) -> np.ndarray | None:
        m = row["metrics"]
        if m["private_safe"] is not True or m["post60_observed"] is not True:
            return None
        y = np.array([m["forward_60_m"], m["lateral_60_m"]], np.float64)
        if not np.isfinite(y).all():
            raise ValueError("finite native calibration outcomes required")
        return y

    baseline = output(rows[0])
    if baseline is None:
        raise ValueError("safe observed calibration anchor required")
    j = np.zeros((2, 4))
    lower = []
    upper = []
    used: list[tuple[str, ...]] = []
    margin = 0.25 - 1e-7
    for axis, (_, hg) in enumerate(MOTOR_AXES):
        negative, positive = rows[axis * 2 + 1 : axis * 2 + 3]
        for row, sign in [(negative, -1), (positive, 1)]:
            raw = row["motor_plan"]
            delta = raw["delta_hg"]
            expected = [0.0] * 29
            expected[hg] = float(sign * 0.15)
            if (
                tuple(delta) != tuple(expected)
                or raw["origin_counter"] != 0
                or raw["peak_control"] != 24
                or raw["start_control"] != rows[0]["metrics"]["first_foot_control"] - 25
            ):
                raise ValueError("calibration differs from declared axis/window")
        minus, plus = output(negative), output(positive)
        if minus is None and plus is None:
            raise ValueError("no safe response data for a declared axis")
        if minus is not None and plus is not None:
            j[:, axis] = (plus - minus) / 0.3
            used.append((negative["name"], positive["name"]))
        elif plus is not None:
            j[:, axis] = (plus - baseline) / 0.15
            used.append((positive["name"],))
        else:
            assert minus is not None
            j[:, axis] = (baseline - minus) / 0.15
            used.append((negative["name"],))
        lower.append(-margin if minus is not None else 0.0)
        upper.append(margin if plus is not None else 0.0)
    return MotorResponseModel(
        tuple(tuple(float(v) for v in row) for row in j),
        tuple(lower),
        tuple(upper),
        source_hash,
        tuple(used),
    )


def propose_response_motor_plan(
    model: MotorResponseModel,
    parent: dict[str, Any],
    *,
    origin_counter: int,
    donor_policy_hash: str,
    parent_targets: np.ndarray,
    limits_hg: np.ndarray,
) -> tuple[CFtrackMotorResidualPlan, dict[str, Any]]:
    if (
        type(model) is not MotorResponseModel
        or type(origin_counter) is not int
        or origin_counter < 0
        or parent.get("private_safe") is not True
        or parent.get("post60_observed") is not True
    ):
        raise ValueError("fitted model and current safe own-parent prediction required")
    first = parent["first_foot_control"]
    if type(first) is not int or not 25 <= first <= 99 or origin_counter + first + 60 > 299:
        raise ValueError("complete current control window inside original deadline required")
    baseline = np.array([parent["forward_60_m"], parent["lateral_60_m"]], np.float64)
    j = np.array(model.jacobian, np.float64)
    lo = np.array(model.lower, np.float64)
    hi = np.array(model.upper, np.float64)
    if (
        baseline.shape != (2,)
        or j.shape != (2, 4)
        or lo.shape != (4,)
        or hi.shape != (4,)
        or not all(np.isfinite(x).all() for x in (baseline, j, lo, hi))
        or (lo > hi).any()
        or (abs(lo) > 0.25).any()
        or (abs(hi) > 0.25).any()
        or baseline[0] <= 0
    ):
        raise ValueError("finite local model and original .25rad box required")
    if (
        type(parent_targets) is not np.ndarray
        or parent_targets.shape != (160, 29)
        or parent_targets.dtype != np.float64
        or not np.isfinite(parent_targets).all()
        or type(limits_hg) is not np.ndarray
        or limits_hg.shape != (29, 2)
        or limits_hg.dtype != np.float64
        or not np.isfinite(limits_hg).all()
        or (limits_hg[:, 0] >= limits_hg[:, 1]).any()
    ):
        raise ValueError("current own-parent NN targets and physical HG29 limits required")
    fraction = _c2_offsets(np.ones(29), tuple(range(29)), 24)[0][:, 0]
    changed = fraction > 0
    for axis, (_, hg) in enumerate(MOTOR_AXES):
        target_window = parent_targets[first - 25 : first + 24, hg][changed]
        lo[axis] = max(
            lo[axis], float(((limits_hg[hg, 0] + 1e-7 - target_window) / fraction[changed]).max())
        )
        hi[axis] = min(
            hi[axis], float(((limits_hg[hg, 1] - 1e-7 - target_window) / fraction[changed]).min())
        )
    if (lo > hi).any():
        raise ValueError("no feasible parent-linearized physical target envelope")
    # New planning margin, not a change to the frozen H300 exam: x>=1.2,
    # |y|<=.2*x. Fixed weights/regularization, no amplitude or seed sweep.
    weights = np.diag([1.0, 4.0])
    target = np.array([max(1.5, float(baseline[0])), 0.0])
    h = j.T @ weights @ j + 0.1 * np.eye(4)
    g = j.T @ weights @ (baseline - target)
    a = np.r_[np.eye(4), -np.eye(4), np.array([-j[0], j[1] - 0.2 * j[0], -j[1] - 0.2 * j[0]])]
    b = np.r_[
        hi,
        -lo,
        np.array(
            [baseline[0] - 1.2, 0.2 * baseline[0] - baseline[1], 0.2 * baseline[0] + baseline[1]]
        ),
    ]
    delta, certificate = solve_small_strict_qp(h, g, a, b)
    full = [0.0] * 29
    for value, (_, hg) in zip(delta, MOTOR_AXES, strict=True):
        full[hg] = float(value)
    plan = CFtrackMotorResidualPlan(
        origin_counter, origin_counter + first - 25, 24, tuple(full), donor_policy_hash
    )
    plan.offsets()
    predicted = baseline + j @ delta
    return plan, dict(
        source_pilot_hash=model.source_pilot_hash,
        fitted_jacobian=model.jacobian,
        fitted_arms=model.fitted_arms,
        parent_outcome_xy=baseline.tolist(),
        predicted_outcome_xy=predicted.tolist(),
        planning_minimum_forward_m=1.2,
        planning_ratio_margin=0.2,
        objective_weights=[1.0, 4.0],
        quadratic_regularization=0.1,
        effective_lower_bounds=lo.tolist(),
        effective_upper_bounds=hi.tolist(),
        parent_target_envelope_not_guarantee_of_candidate_nn_targets=True,
        calibration_amplitude_rad=0.15,
        proposal_extrapolates_sampled_amplitude=bool(abs(delta).max() > 0.15),
        joint_coupling_unmeasured=True,
        private_native_validation_required=True,
        neural_optimizer_updates=0,
        qp_certificate=certificate,
        hardware_authorized=False,
    )
