"""Reference contact-label consistency diagnostics; no automatic relabeling."""

from typing import Any

import numpy as np

from rosclaw_soccer.rsi.candidate_motion23_mapping import candidate_map_motion23
from rosclaw_soccer.rsi.native_velocity_transfer import NativeVelocityLayout
from rosclaw_soccer.rsi.shared_target_forecast import compiled_model_hash


def reference_ankle_positions(
    model: Any, fields: dict[str, np.ndarray], *, expected_model_hash: str
) -> np.ndarray:
    """Return source-frame ankle centers in scratch kinematics, never dynamics."""
    import mujoco

    if (
        type(fields) is not dict
        or not {"dof", "root_rot", "root_trans_offset"} <= fields.keys()
        or compiled_model_hash(model) != expected_model_hash
    ):
        raise ValueError("pinned body and complete reference fields required")
    mapped = candidate_map_motion23(fields["dof"], fields["root_rot"])
    joints, rotation = mapped["dof29"], mapped["root_rot_wxyz"]
    positions = fields["root_trans_offset"]
    if (
        type(positions) is not np.ndarray
        or positions.shape != (len(joints), 3)
        or positions.dtype.kind != "f"
        or not np.isfinite(positions).all()
    ):
        raise ValueError("finite root positions required")
    layout = NativeVelocityLayout(model, robot_prefix="a/", ball_body_name="soccer_ball")
    ankles = [int(model.body(f"a/{side}_ankle_roll_link").id) for side in ("left", "right")]
    data = mujoco.MjData(model)
    data.qpos[layout.bq : layout.bq + 7] = [100, 100, 0.11, 1, 0, 0, 0]
    result = np.empty((len(joints), 2, 3), dtype=np.float64)
    for i in range(len(joints)):
        data.qpos[layout.rq : layout.rq + 3] = positions[i]
        data.qpos[layout.rq + 3 : layout.rq + 7] = rotation[i]
        data.qpos[layout.qi] = joints[i]
        mujoco.mj_kinematics(model, data)
        result[i] = data.xpos[ankles]
    if data.time != 0 or not np.isfinite(result).all():
        raise ValueError("finite kinematics without physical integration required")
    return result


def contact_label_motion_diagnostic(
    ankle_positions: np.ndarray,
    claimed_contacts: np.ndarray,
    *,
    fps: float,
    speed_threshold_m_s: float = 0.5,
) -> dict[str, Any]:
    """Describe fast ankle centers during claimed support, not contact truth."""
    if (
        type(ankle_positions) is not np.ndarray
        or ankle_positions.ndim != 3
        or ankle_positions.shape[1:] != (2, 3)
        or not 3 <= len(ankle_positions) <= 60000
        or ankle_positions.dtype.kind != "f"
        or not np.isfinite(ankle_positions).all()
        or type(claimed_contacts) is not np.ndarray
        or claimed_contacts.shape != ankle_positions.shape[:2]
        or claimed_contacts.dtype != np.bool_
        or not claimed_contacts.any()
        or type(fps) is not float
        or not np.isfinite(fps)
        or not 1 <= fps <= 240
        or type(speed_threshold_m_s) is not float
        or not np.isfinite(speed_threshold_m_s)
        or not 0 < speed_threshold_m_s <= 5
    ):
        raise ValueError("finite time-ordered ankles, labels and declared frame rate required")
    velocity = np.gradient(ankle_positions, 1.0 / fps, axis=0)
    horizontal = np.linalg.norm(velocity[:, :, :2], axis=2)
    support_speed = horizontal[claimed_contacts]
    return dict(
        source_frames=len(ankle_positions),
        claimed_contact_samples=len(support_speed),
        claimed_contact_ankle_speed_p95_m_s=float(np.quantile(support_speed, 0.95)),
        claimed_contact_fast_fraction=float((support_speed > speed_threshold_m_s).mean()),
        speed_threshold_m_s=speed_threshold_m_s,
        frame_rate_hz=fps,
        reference_ankle_center_speed_is_not_contact_point_slip=True,
        labels_automatically_changed=False,
        source_contact_truth_not_proven=True,
        integrated_native_steps=0,
        physical_execution_qualified=False,
        hardware_authorized=False,
    )
