"""Scratch-FK multiframe experiment, not motion control or learned policy."""

from typing import Any

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix


def fit_temporal_contact(
    model: Any,
    layout: Any,
    qposes: np.ndarray,
    *,
    foot: str,
    event: int,
    max_evaluations: int = 50,
) -> dict[str, Any]:
    """Jointly fit a window, with fixed root, ball, support leg and endpoints.

    Residuals use native signed distances and adjacent correction differences.
    Final hard diagnostic gates are independent of optimizer termination; even
    passage establishes only sampled kinematics, never dynamics or force.
    """
    import mujoco

    if (
        layout.model is not model
        or foot not in ("left", "right")
        or type(qposes) is not np.ndarray
        or qposes.ndim != 2
        or qposes.shape[1] != model.nq
        or not 5 <= len(qposes) <= 101
        or not np.isfinite(qposes).all()
        or type(event) is not int
        or not 1 <= event < len(qposes) - 1
        or type(max_evaluations) is not int
        or not 1 <= max_evaluations <= 1000
    ):
        raise ValueError("bounded finite trajectory and declared interior event required")
    n = len(qposes)
    active = np.arange(0, 6) if foot == "left" else np.arange(6, 12)
    joints = np.asarray(layout.joints)[active]
    if not np.all(model.jnt_limited[joints]):
        raise ValueError("native joint limits required")
    original = qposes[:, layout.qi[active]]
    limits = model.jnt_range[joints]
    lower = np.maximum(-0.35, limits[:, 0] - original[1:-1])
    upper = np.minimum(0.35, limits[:, 1] - original[1:-1])
    if np.any(lower > 0) or np.any(upper < 0) or np.any(lower >= upper):
        raise ValueError("source active joints outside native limits")
    ball = np.flatnonzero(model.geom_bodyid == layout.ball)
    if len(ball) != 1 or model.geom_type[ball[0]] != mujoco.mjtGeom.mjGEOM_SPHERE:
        raise ValueError("one native sphere required")
    foot_body = int(model.body(layout.robot_prefix + foot + "_ankle_roll_link").id)
    foot_geoms = np.flatnonzero(
        (model.geom_bodyid == foot_body)
        & ((model.geom_contype != 0) | (model.geom_conaffinity != 0))
    )
    shins = [int(model.geom(layout.robot_prefix + side + "_shin").id) for side in ("left", "right")]
    if len(foot_geoms) == 0:
        raise ValueError("native colliding foot geometry required")
    data = mujoco.MjData(model)

    def unpack(flat: np.ndarray) -> np.ndarray:
        return np.vstack((np.zeros((1, 6)), flat.reshape(n - 2, 6), np.zeros((1, 6))))

    def distances(offset: np.ndarray) -> np.ndarray:
        out = np.empty((n, 3))
        for i in range(n):
            data.qpos[:] = qposes[i]
            data.qpos[layout.qi[active]] += offset[i]
            mujoco.mj_kinematics(model, data)
            mujoco.mj_comPos(model, data)

            def gap(geom: int) -> float:
                return float(mujoco.mj_geomDistance(model, data, geom, int(ball[0]), 2.0, None))

            out[i] = [min(gap(int(g)) for g in foot_geoms), *(gap(g) for g in shins)]
        return out

    def residual(flat: np.ndarray) -> np.ndarray:
        offsets = unpack(flat)
        gaps = distances(offsets)
        # 1 mm positive margin on each foot, 5 mm on shins; event target is
        # diagnostic proximity only and not evidence of a purposeful touch.
        geometry = np.maximum(0.0, np.array([0.001, 0.005, 0.005]) - gaps) * 200
        frame_cost = np.column_stack((geometry, offsets * 0.15)).ravel()
        step = np.diff(offsets, axis=0)
        accel = np.diff(step, axis=0)
        rate_penalty = np.maximum(0.0, np.abs(step) - 0.08) * 20
        return np.concatenate(
            (
                frame_cost,
                (step * 0.5).ravel(),
                (accel * 0.5).ravel(),
                rate_penalty.ravel(),
                np.array([(gaps[event, 0] - 0.004) * 200]),
            )
        )

    # Each geometry row depends only on its frame; acceleration on three.
    rows = 9 * n + 6 * (n - 1) + 6 * (n - 2) + 6 * (n - 1) + 1
    sparsity = lil_matrix((rows, 6 * (n - 2)), dtype=np.int8)

    def bind(row: int, frames: tuple[int, ...], width: int) -> None:
        for frame in frames:
            if 0 < frame < n - 1:
                sparsity[row : row + width, 6 * (frame - 1) : 6 * frame] = 1

    for i in range(n):
        bind(9 * i, (i,), 9)
    cursor = 9 * n
    for i in range(n - 1):
        bind(cursor + 6 * i, (i, i + 1), 6)
    cursor += 6 * (n - 1)
    for i in range(n - 2):
        bind(cursor + 6 * i, (i, i + 1, i + 2), 6)
    cursor += 6 * (n - 2)
    for i in range(n - 1):
        bind(cursor + 6 * i, (i, i + 1), 6)
    bind(rows - 1, (event,), 1)
    before = distances(np.zeros((n, 6)))
    result = least_squares(
        residual,
        np.zeros(6 * (n - 2)),
        bounds=(lower.ravel(), upper.ravel()),
        jac_sparsity=sparsity.tocsr(),
        max_nfev=max_evaluations,
    )
    offsets = unpack(result.x)
    after = distances(offsets)
    max_step = float(np.max(np.abs(np.diff(offsets, axis=0))))
    gates = dict(
        no_foot_penetration=bool(np.min(after[:, 0]) >= -0.001),
        no_shin_penetration=bool(np.min(after[:, 1:]) >= -0.001),
        event_foot_near_ball=bool(-0.001 <= after[event, 0] <= 0.01),
        offset_rate_bound=bool(max_step <= 0.08),
        optimizer_converged=bool(result.success),
    )
    return dict(
        status="SAMPLED_KINEMATICS_ONLY" if all(gates.values()) else "NOT_QUALIFIED",
        foot=foot,
        gates=gates,
        before_gaps_m=before.tolist(),
        after_gaps_m=after.tolist(),
        active_leg_offsets_rad=offsets.tolist(),
        max_offset_step_rad=max_step,
        evaluations=int(result.nfev),
        evaluation_budget=max_evaluations,
        optimizer_message=str(result.message),
        physics_steps=0,
        physically_qualified=False,
        promotion_authorized=False,
        hardware_authorized=False,
    )
