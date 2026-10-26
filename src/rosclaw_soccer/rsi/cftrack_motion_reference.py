"""Validate a complete research motion before allocating a native SIM episode.

This is reference data, never measured ball state or executable joint targets.
No slicing, time warping, contact relabeling or radius changes are performed.
"""

from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray


def load_cftrack_motion(path: Path, *, maximum_frames: int = 540) -> dict[str, NDArray[Any]]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("ordinary complete motion file required")
    if type(maximum_frames) is not int or maximum_frames < 9:
        raise ValueError("finite declared motion frame budget required")
    shapes = {
        "base_pos_w": (3,),
        "base_quat_w": (4,),
        "joint_pos": (29,),
        "joint_vel": (29,),
        "body_pos_w": (39, 3),
        "body_quat_w": (39, 4),
        "body_lin_vel_w": (39, 3),
        "body_ang_vel_w": (39, 3),
        "ee_pos_w": (5, 3),
        "ee_quat_w": (5, 4),
        "object_pos_w": (3,),
        "object_quat_w": (4,),
        "object_lin_vel_w": (3,),
        "object_ang_vel_w": (3,),
        "table1_pos_w": (3,),
        "table2_pos_w": (3,),
        "contact_info": (4, 1),
    }
    with np.load(path, allow_pickle=False) as archive:
        if not set(shapes).union({"fps"}).issubset(archive.files):
            raise ValueError("complete source motion schema required")
        data = {key: archive[key].copy() for key in (*shapes, "fps")}
    fps = data["fps"]
    if fps.shape != (1,) or fps.dtype.kind != "f" or not np.array_equal(fps, [50.0]):
        raise ValueError("original 50 Hz motion required; no implicit resampling")
    joints = data["joint_pos"]
    if joints.ndim != 2 or not 9 <= len(joints) <= maximum_frames:
        raise ValueError("complete motion exceeds declared frame budget")
    frames = len(joints)
    for key, tail in shapes.items():
        value = data[key]
        if (
            value.shape != (frames, *tail)
            or value.dtype.kind != "f"
            or not np.isfinite(value).all()
        ):
            raise ValueError("finite aligned floating motion arrays required: " + key)
        if "quat" in key and not np.allclose(np.linalg.norm(value, axis=-1), 1, atol=1e-3, rtol=0):
            raise ValueError("unit wxyz motion quaternions required: " + key)
    if not np.isin(data["contact_info"], [0.0, 1.0]).all():
        raise ValueError("binary original contact flags required")
    # The author's loader/reset takes its root from body index zero. Redundant
    # base arrays can differ. Retain both rather than silently reconciling them.
    for value in data.values():
        value.flags.writeable = False
    return data
