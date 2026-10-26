"""Unqualified research hypothesis for converting a 23-DoF reference to G1-29.

The fixed hypothesis follows the layout used by a public reference converter;
it is not provenance proof, dynamic retargeting, a motor policy or execution.
"""

from typing import Any

import numpy as np


def candidate_map_motion23(dof: np.ndarray, root_rot_xyzw: np.ndarray) -> dict[str, Any]:
    """Insert six zero wrists and reorder XYZW to WXYZ without temporal resampling."""
    if (
        type(dof) is not np.ndarray
        or type(root_rot_xyzw) is not np.ndarray
        or dof.ndim != 2
        or dof.shape[1] != 23
        or not 2 <= len(dof) <= 60000
        or root_rot_xyzw.shape != (len(dof), 4)
        or dof.dtype.kind != "f"
        or root_rot_xyzw.dtype.kind != "f"
        or not np.isfinite(dof).all()
        or not np.isfinite(root_rot_xyzw).all()
        or not np.allclose(np.linalg.norm(root_rot_xyzw, axis=1), 1.0, rtol=0, atol=1e-4)
    ):
        raise ValueError("finite 23-joint sequence and unit XYZW quaternion required")
    dof29 = np.zeros((len(dof), 29), dtype=dof.dtype)
    dof29[:, :19] = dof[:, :19]
    dof29[:, 22:26] = dof[:, 19:23]
    quaternion = root_rot_xyzw[:, [3, 0, 1, 2]].copy()
    return dict(
        dof29=dof29,
        root_rot_wxyz=quaternion,
        zero_filled_wrist_indices=(19, 20, 21, 26, 27, 28),
        input_joint_order_is_assumed=True,
        input_quaternion_convention_is_assumed=True,
        source_provenance_verified=False,
        body_compatibility_verified=False,
        physical_execution_qualified=False,
        hardware_authorized=False,
    )
