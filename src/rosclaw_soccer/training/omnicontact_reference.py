"""Explicit, read-only pose references from externally authenticated OmniContact.

CC BY-NC research data stays external. This mapper is not a motor policy, a
receiving label generator, or evidence of dynamically feasible tracking.
"""

import math
import re
from typing import Any

import numpy as np

# Official lab2mj permutation, independently checked by CPU MuJoCo FK.
LAB_TO_MUJOCO = (
    0,
    3,
    6,
    9,
    13,
    17,
    1,
    4,
    7,
    10,
    14,
    18,
    2,
    5,
    8,
    11,
    15,
    19,
    21,
    23,
    25,
    27,
    12,
    16,
    20,
    22,
    24,
    26,
    28,
)
# Loader calls 37/38 "wrists"; FK identifies the actual palm origins.
BODY_INDICES = {
    "pelvis": 0,
    "torso_link": 11,
    "left_ankle_roll_link": 25,
    "right_ankle_roll_link": 26,
    "left_palm_link": 37,
    "right_palm_link": 38,
}


def omnicontact_pose_reference(
    *,
    joint_pos: np.ndarray,
    body_pos_w: np.ndarray,
    body_quat_w: np.ndarray,
    fps: float,
    source_hash: str,
) -> dict[str, Any]:
    """Map supplied arrays without guessing order, velocity, contact or support.

    The caller must authenticate the external file and its declared split.
    Hash syntax validation here does not authenticate data or grant its license.
    Returned copies cannot mutate the supplied arrays and are write-protected.
    """
    if (
        not isinstance(source_hash, str)
        or re.fullmatch(r"sha256:[0-9a-f]{64}", source_hash) is None
        or type(fps) not in (int, float)
        or not math.isfinite(fps)
        or not 1 <= fps <= 1000
        or not isinstance(joint_pos, np.ndarray)
        or joint_pos.ndim != 2
        or not 2 <= len(joint_pos) <= 100_000
    ):
        raise ValueError("bounded measured pose sequence and explicit provenance required")
    count = len(joint_pos)
    for values, shape in (
        (joint_pos, (count, 29)),
        (body_pos_w, (count, 39, 3)),
        (body_quat_w, (count, 39, 4)),
    ):
        if (
            not isinstance(values, np.ndarray)
            or values.shape != shape
            or values.dtype.kind not in "fi"
            or not np.isfinite(values).all()
        ):
            raise ValueError("finite aligned OmniContact pose arrays required")
    if np.any(np.abs(np.linalg.norm(body_quat_w, axis=-1) - 1) > 0.01):
        raise ValueError("source body quaternions must be normalized wxyz")

    def readonly(values: np.ndarray) -> np.ndarray:
        copied = np.array(values, copy=True)
        copied.setflags(write=False)
        return copied

    return {
        "schema": "soccer.omnicontact_pose_reference.v1",
        "source_hash": source_hash,
        "source_authenticated_here": False,
        "split_checked_here": False,
        "fps": float(fps),
        "time_sec": readonly(np.arange(count, dtype=np.float64) / fps),
        "joint_position_mujoco_rad": readonly(joint_pos[:, LAB_TO_MUJOCO]),
        "body_position_m": {
            name: readonly(body_pos_w[:, index]) for name, index in BODY_INDICES.items()
        },
        "body_quaternion_wxyz": {
            name: readonly(body_quat_w[:, index]) for name, index in BODY_INDICES.items()
        },
        "joint_order": "official_g1_29dof_mujoco",
        "support_state": None,
        "contact_force": None,
        "receiving_success": None,
        "dynamic_feasibility_verified": False,
        "policy_action": False,
        "research_only_noncommercial_source": True,
        "promotion_authorized": False,
    }
