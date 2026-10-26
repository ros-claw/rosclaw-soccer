"""Unqualified reference-only static geometry audit; never advances simulation."""

from typing import Any

import numpy as np

from rosclaw_soccer.rsi.candidate_motion23_mapping import candidate_map_motion23
from rosclaw_soccer.rsi.native_velocity_transfer import NativeVelocityLayout
from rosclaw_soccer.rsi.shared_target_forecast import compiled_model_hash


def audit_candidate_fk(
    model: Any, fields: dict[str, np.ndarray], *, expected_model_hash: str
) -> dict[str, Any]:
    """Check a named-body candidate at source frames, not executable motions."""
    import mujoco

    if (
        type(fields) is not dict
        or not {"dof", "root_rot", "root_trans_offset"} <= fields.keys()
        or type(expected_model_hash) is not str
        or compiled_model_hash(model) != expected_model_hash
    ):
        raise ValueError("explicit pinned simulation body and candidate fields required")
    mapped = candidate_map_motion23(fields["dof"], fields["root_rot"])
    joints, rotation = mapped["dof29"], mapped["root_rot_wxyz"]
    positions = fields["root_trans_offset"]
    if (
        type(positions) is not np.ndarray
        or positions.shape != (len(joints), 3)
        or positions.dtype.kind != "f"
        or not np.isfinite(positions).all()
    ):
        raise ValueError("finite root positions at every source frame required")
    layout = NativeVelocityLayout(model, robot_prefix="a/", ball_body_name="soccer_ball")
    floor = int(model.geom("floor").id)
    data = mujoco.MjData(model)
    data.qpos[layout.bq : layout.bq + 7] = [100, 100, 0.11, 1, 0, 0, 0]
    limits = model.jnt_range[layout.joints]
    limited = model.jnt_limited[layout.joints].astype(bool)
    excess = np.maximum(np.maximum(limits[:, 0] - joints, joints - limits[:, 1]), 0) * limited
    floor_distances: list[float] = []
    for i in range(len(joints)):
        data.qpos[layout.rq : layout.rq + 3] = positions[i]
        data.qpos[layout.rq + 3 : layout.rq + 7] = rotation[i]
        data.qpos[layout.qi] = joints[i]
        mujoco.mj_forward(model, data)
        floor_distances.extend(
            float(c.dist) for c in data.contact if floor in (int(c.geom[0]), int(c.geom[1]))
        )
    if data.time != 0:
        raise ValueError("reference geometry audit must never integrate")
    return dict(
        source_frames=len(joints),
        joint_limit_violation_frames=int((excess.max(axis=1) > 1e-6).sum()),
        maximum_joint_limit_excess_rad=float(excess.max()),
        root_below_065m_frames=int((positions[:, 2] < 0.65).sum()),
        minimum_floor_contact_distance_m=min(floor_distances) if floor_distances else None,
        floor_contact_count=len(floor_distances),
        reference_mapping_assumed=True,
        source_root_height_unmodified=True,
        scratch_ball_far_away_only_for_geometry=True,
        integrated_native_steps=0,
        physical_execution_qualified=False,
        hardware_authorized=False,
    )
