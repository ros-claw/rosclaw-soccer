"""SIM reference-only foot geometry and global coordinate-offset diagnostic."""

from typing import Any, cast

import numpy as np

from rosclaw_soccer.rsi.candidate_motion23_mapping import candidate_map_motion23
from rosclaw_soccer.rsi.native_velocity_transfer import NativeVelocityLayout
from rosclaw_soccer.rsi.shared_target_forecast import compiled_model_hash


def support_distances(
    model: Any, fields: dict[str, np.ndarray], *, expected_model_hash: str
) -> tuple[np.ndarray, np.ndarray]:
    """Measure source-frame capsule/plane gaps; labels are unverified source claims."""
    import mujoco

    if (
        type(fields) is not dict
        or not {"dof", "root_rot", "root_trans_offset", "contact_mask"} <= fields.keys()
        or type(expected_model_hash) is not str
        or compiled_model_hash(model) != expected_model_hash
    ):
        raise ValueError("pinned candidate model and complete reference fields required")
    mapped = candidate_map_motion23(fields["dof"], fields["root_rot"])
    joints, rotation = mapped["dof29"], mapped["root_rot_wxyz"]
    positions, labels = fields["root_trans_offset"], fields["contact_mask"]
    if (
        type(positions) is not np.ndarray
        or type(labels) is not np.ndarray
        or positions.shape != (len(joints), 3)
        or labels.shape != (len(joints), 2)
        or positions.dtype.kind != "f"
        or labels.dtype.kind not in "fiu"
        or not np.isfinite(positions).all()
        or not np.isfinite(labels).all()
        or not np.isin(labels, [0, 1]).all()
    ):
        raise ValueError("finite source positions and binary claimed foot contacts required")
    layout = NativeVelocityLayout(model, robot_prefix="a/", ball_body_name="soccer_ball")
    floor = int(model.geom("floor").id)
    feet = [
        [int(model.geom(f"a/{side}_foot{i}_collision").id) for i in range(1, 8)]
        for side in ("left", "right")
    ]
    if model.geom_type[floor] != int(mujoco.mjtGeom.mjGEOM_PLANE) or any(
        model.geom_type[g] != int(mujoco.mjtGeom.mjGEOM_CAPSULE) for foot in feet for g in foot
    ):
        raise ValueError("explicit plane and 14 capsule foot collision geometries required")
    data = mujoco.MjData(model)
    data.qpos[layout.bq : layout.bq + 7] = [100, 100, 0.11, 1, 0, 0, 0]
    distances = np.empty((len(joints), 2), dtype=np.float64)
    for i in range(len(joints)):
        data.qpos[layout.rq : layout.rq + 3] = positions[i]
        data.qpos[layout.rq + 3 : layout.rq + 7] = rotation[i]
        data.qpos[layout.qi] = joints[i]
        mujoco.mj_kinematics(model, data)
        if not np.allclose(data.geom_xmat[floor].reshape(3, 3)[:, 2], [0, 0, 1], atol=1e-9):
            raise ValueError("fixed upward plane normal required for vertical distance")
        plane_z = data.geom_xpos[floor, 2]
        for side, foot in enumerate(feet):
            distances[i, side] = min(
                float(
                    data.geom_xpos[g, 2]
                    - model.geom_size[g, 0]
                    - model.geom_size[g, 1] * abs(data.geom_xmat[g].reshape(3, 3)[2, 2])
                    - plane_z
                )
                for g in foot
            )
    if data.time != 0 or not np.isfinite(distances).all():
        raise ValueError("finite geometry without native integration required")
    return distances, labels.astype(bool, copy=True)


def fit_support_offset(train_distances: np.ndarray, train_labels: np.ndarray) -> dict[str, Any]:
    """Fit one least-squares z-coordinate hypothesis; never alter a reference."""
    values = _selected(train_distances, train_labels)
    return dict(
        fitted_offset_m=-float(values.mean()),
        labeled_train_samples=len(values),
        source_contact_labels_assumed=True,
        candidate_mapping_assumed=True,
        physical_execution_qualified=False,
        hardware_authorized=False,
    )


def score_support_offset(
    distances: np.ndarray, labels: np.ndarray, *, fitted_offset_m: float
) -> dict[str, Any]:
    """Score only supplied frames, without adjusting or fitting the offset."""
    values = _selected(distances, labels)
    if type(fitted_offset_m) not in (float, int) or not np.isfinite(fitted_offset_m):
        raise ValueError("finite previously fitted offset required")
    residual = np.abs(values + fitted_offset_m)
    return dict(
        labeled_samples=len(values),
        mean_absolute_residual_m=float(residual.mean()),
        p95_absolute_residual_m=float(np.quantile(residual, 0.95)),
        fraction_over_002m=float((residual > 0.02).mean()),
        physical_execution_qualified=False,
    )


def _selected(distances: np.ndarray, labels: np.ndarray) -> np.ndarray:
    if (
        type(distances) is not np.ndarray
        or type(labels) is not np.ndarray
        or distances.ndim != 2
        or distances.shape[1] != 2
        or labels.shape != distances.shape
        or labels.dtype != np.bool_
        or not np.isfinite(distances).all()
        or not labels.any()
    ):
        raise ValueError("finite paired distances with explicit nonempty boolean labels required")
    return cast(np.ndarray, distances[labels])
