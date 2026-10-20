"""Sagittal reflection of canonical reference data, never a body-state reset.

This is a kinematic data augmentation, not a certified left-foot controller.
The caller must qualify the named native body's FK, limits and closed-loop
tracking separately. Quarantine wrists from the OWN measured entry afterwards.
"""

import numpy as np

from rosclaw_soccer.providers.g1.joint_contract import G1_DDS_JOINT_NAMES
from rosclaw_soccer.providers.g1.sonic_reference_preparation import _pose_array


def mirror_sagittal_reference(poses: np.ndarray) -> np.ndarray:
    """Reflect about world y=0, including axial wxyz orientation and named joints.

    Pitch, knee and elbow rotations retain sign. Roll/yaw reverse sign and
    left/right channels exchange. Applying twice returns the original data.
    This G1 convention must not be generalized to arbitrary robot joint axes.
    """
    source = _pose_array(poses, rows=True)
    reflected = source.copy()
    reflected[:, 1] *= -1
    reflected[:, [4, 6]] *= -1
    for index, name in enumerate(G1_DDS_JOINT_NAMES):
        counterpart = (
            "right_" + name[5:]
            if name.startswith("left_")
            else "left_" + name[6:]
            if name.startswith("right_")
            else name
        )
        sign = -1 if "_roll_" in name or "_yaw_" in name else 1
        reflected[:, 7 + index] = sign * source[:, 7 + G1_DDS_JOINT_NAMES.index(counterpart)]
    return reflected
