"""Explicit foundation-relative target increments, NOT measured-pose IK.

The original policy's target/current tracking error is NOT bounded by this
proposal. Authority is on the additional target increment only. Jacobians
must still be measured at actual q; native forecasts test physical response.
"""

from typing import Any

import numpy as np

from rosclaw_soccer.physics.bounded_joint_tasks import LinearJointTask


def solve_foundation_delta(
    current: np.ndarray,
    nominal: np.ndarray,
    previous_delta: np.ndarray,
    limits: np.ndarray,
    tasks: tuple[LinearJointTask, ...],
) -> tuple[np.ndarray | None, dict[str, Any]]:
    for a, shape in (
        (current, (29,)),
        (nominal, (29,)),
        (previous_delta, (29,)),
        (limits, (29, 2)),
    ):
        if a.shape != shape or a.dtype != np.float64 or not np.isfinite(a).all():
            raise ValueError("finite actual current/nominal/residual/physical limits required")
    if np.any(limits[:, 0] >= limits[:, 1]) or np.any(abs(previous_delta) > 0.120001):
        raise ValueError("ordered physical limits and bounded previous target increment required")
    if not 1 <= len(tasks) <= 8 or any(
        not isinstance(t, LinearJointTask)
        or t.jacobian.shape != (3, 29)
        or t.displacement_m.shape != (3,)
        or not np.isfinite(t.jacobian).all()
        or not np.isfinite(t.displacement_m).all()
        or np.linalg.norm(t.displacement_m) > 0.25
        or not 0 < t.cost <= 100
        for t in tasks
    ):
        raise ValueError("bounded current-state incremental material-point tasks required")
    lower = np.maximum.reduce(
        (limits[:12, 0] - nominal[:12], np.full(12, -0.12), previous_delta[:12] - 0.03)
    )
    upper = np.minimum.reduce(
        (limits[:12, 1] - nominal[:12], np.full(12, 0.12), previous_delta[:12] + 0.03)
    )
    report = dict(
        nominal_tracking_error_rad=float(np.max(abs(nominal[:12] - current[:12]))),
        measured_pose_ik=False,
        maximum_increment_rad=0.12,
        candidate_increment_step_cap_rad=0.03,
        fallback_to_parent_resets_increment=True,
        physical_response_proven=False,
    )
    if np.any(lower > upper):
        return None, dict(report, status="ABSTAIN_INCREMENT_BOUND_INTERSECTION")
    matrix = np.vstack(
        [*(np.sqrt(t.cost) * t.jacobian[:, :12] for t in tasks), np.sqrt(0.1) * np.eye(12)]
    )
    desired = np.concatenate([*(np.sqrt(t.cost) * t.displacement_m for t in tasks), np.zeros(12)])
    fixed = lower == upper
    delta = np.zeros(12)
    delta[fixed] = lower[fixed]
    if np.any(~fixed):
        from scipy.optimize import lsq_linear  # type: ignore[import-untyped]

        solved = lsq_linear(
            matrix[:, ~fixed],
            desired - matrix[:, fixed] @ delta[fixed],
            bounds=(lower[~fixed], upper[~fixed]),
            method="bvls",
            tol=1e-10,
            max_iter=300,
        )
        if not solved.success or not np.isfinite(solved.x).all():
            return None, dict(report, status="ABSTAIN_INCREMENT_SOLVER")
        delta[~fixed] = solved.x
    result = nominal.copy()
    result[:12] = np.clip(nominal[:12] + delta, nominal[:12] + lower, nominal[:12] + upper)
    return result, dict(report, status="FOUNDATION_INCREMENT_PROPOSAL_NOT_PHYSICALLY_QUALIFIED")


def incremental_task_candidate(
    layout: Any,
    donor: Any,
    nominal: np.ndarray,
    previous: np.ndarray,
    surfaces: dict[str, Any],
    *,
    contact: bool,
) -> tuple[np.ndarray | None, str]:
    feet = [(n, s) for n, s in surfaces.items() if n.endswith("foot") and s["available"]]
    if len(feet) != 2:
        return None, "ABSTAIN_NO_TWO_FOOT_SURFACES"
    active = min(feet, key=lambda item: item[1]["gap_m"])[0]
    tasks = []
    for name, foot in feet:
        correction = np.zeros(3)
        if contact and name == active:
            gap_vector = foot["ball_point"] - foot["world_point"]
            correction[:] = [
                0.02,
                np.clip(gap_vector[1], -0.03, 0.03),
                np.clip(gap_vector[2], -0.015, 0.015),
            ]
        tasks.append(LinearJointTask(name, foot["jacobian_hg"], correction, 100.0))
    for name, shin in surfaces.items():
        if name.endswith("shin") and shin["available"] and shin["smooth_separated"]:
            correction = -shin["normal_toward_ball"] * float(
                np.clip(0.015 - shin["gap_m"], 0, 0.04)
            )
            tasks.append(LinearJointTask(name, shin["jacobian_hg"], correction, 100.0))
    candidate, report = solve_foundation_delta(
        donor.qpos[layout.qi].copy(),
        nominal,
        previous,
        layout.model.jnt_range[layout.joints].astype(np.float64),
        tuple(tasks),
    )
    return candidate, report["status"]
