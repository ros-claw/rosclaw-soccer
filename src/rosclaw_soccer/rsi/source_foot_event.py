"""Source-only foot forward-motion cue; kinematics, never ball/task labels."""

from dataclasses import dataclass
from typing import Any

import mujoco
import numpy as np

from rosclaw_soccer.providers.g1.joint_contract import G1_DDS_JOINT_NAMES
from rosclaw_soccer.providers.g1.sonic_reference_preparation import _pose_array


@dataclass(frozen=True)
class SourceFootEvent:
    interval: int
    foot: str
    root_relative_forward_speed_mps: float


def forward_foot_event(root_local_foot_positions: np.ndarray) -> SourceFootEvent:
    """First maximum forward ankle-origin speed at declared120Hz.

    This may still be a step rather than a kick. It is a deterministic cue for
    a declared reference window, not foot/ball contact or a successful shot.
    """
    p = root_local_foot_positions
    if (
        type(p) is not np.ndarray
        or p.ndim != 3
        or p.shape[1:] != (2, 3)
        or not 2 <= len(p) <= 10000
        or p.dtype.kind != "f"
        or not np.isfinite(p).all()
    ):
        raise ValueError("finite120Hz root-local left/right ankle positions required")
    velocity = np.diff(p[:, :, 0], axis=0) * 120
    interval, side = np.unravel_index(int(np.argmax(velocity)), velocity.shape)
    speed = float(velocity[interval, side])
    if speed < 0.5:
        raise ValueError("source lacks declared0.5m/s forward foot-motion cue")
    return SourceFootEvent(int(interval), ("left", "right")[int(side)], speed)


def source_foot_event(model: Any, source_poses: np.ndarray) -> SourceFootEvent:
    """Private FK on named29-joint G1 source, zero mj_step or body mutation.

    Root-local ankle velocity avoids calling global translation a leg strike.
    Morphology/ground mismatch and physical tracking still need separate tests.
    The caller authenticates model/source and the declared120Hz time contract.
    """
    poses = _pose_array(source_poses, rows=True)
    if not 2 <= len(poses) <= 10000:
        raise ValueError("bounded source FK duration required")
    if (model.nq, model.nv, model.nu) != (43, 41, 29):
        raise ValueError("qualified named G1-plus-ball model required")
    joints = np.asarray([model.joint(name).id for name in G1_DDS_JOINT_NAMES])
    qi = model.jnt_qposadr[joints]
    feet = [int(model.body(side + "_ankle_roll_link").id) for side in ("left", "right")]
    data = mujoco.MjData(model)
    positions = []
    matrix = np.empty(9)
    for pose in poses:
        data.qpos[:7] = pose[:7]
        data.qpos[qi] = pose[7:]
        mujoco.mj_kinematics(model, data)
        mujoco.mju_quat2Mat(matrix, pose[3:7])
        positions.append((data.xpos[feet] - pose[:3]) @ matrix.reshape(3, 3))
    return forward_foot_event(np.asarray(positions))
