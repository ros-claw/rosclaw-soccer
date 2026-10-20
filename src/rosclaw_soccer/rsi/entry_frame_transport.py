"""Read-only initiation-point transport across motion source frames.

Changing the first source pose changes its spatial frame as well as its time.
This module transforms a PAST entry affordance geometrically. It opens no body,
does not admit a motion, and does not certify dynamic or contact feasibility.
"""

import re
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation  # type: ignore[import-untyped]

from rosclaw_soccer.rsi.ball_entry_teacher import SkillEntryAffordance
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json


@dataclass(frozen=True)
class EntryFrameTransport:
    original_affordance_hash: str
    source_motion_artifact_hash: str
    source_root_trajectory_hash: str
    original_source_frame: int
    target_source_frame: int
    original_ball_offset_xy_m: tuple[float, float]
    transported_ball_offset_xy_m: tuple[float, float]
    source_translation_in_original_heading_m: tuple[float, float, float]
    source_heading_delta_rad: float

    def report(self) -> dict[str, Any]:
        return dict(
            schema="soccer.rsi.entry_frame_transport.v1",
            **asdict(self),
            approximation="SE2_horizontal_point_yaw_transport_not_full_dynamic_initiation_set",
            source_height_change_compensated=False,
            changes_source_reference=False,
            changes_actual_body_or_ball_state=False,
            dynamic_feasibility_proven=False,
            contact_qualified=False,
            new_native_steps=0,
            new_neural_calls=0,
            optimizer_updates=0,
            promotion_authorized=False,
            hardware_authorized=False,
        )

    @property
    def contract_hash(self) -> str:
        return hash_json(self.report())


def transport_entry_frame(
    prior: SkillEntryAffordance,
    source_roots: np.ndarray[Any, Any],
    *,
    original_source_frame: int,
    target_source_frame: int,
    source_motion_artifact_hash: str,
) -> EntryFrameTransport:
    """Keep one source-world point invariant while expressing it in a new frame.

    Roots are namedXYZ/WXYZ only: this is NOT 29-joint motion admission. The
    source file identity and original affordance remain in the report. Source
    quaternions are interpreted by SciPy; no actual body pose is normalized.
    """
    prior.__post_init__()
    roots = source_roots
    if (
        type(roots) is not np.ndarray
        or roots.ndim != 2
        or roots.shape[1] != 7
        or not 2 <= len(roots) <= 10000
        or roots.dtype != np.float64
        or not np.isfinite(roots).all()
        or np.any(abs(np.linalg.norm(roots[:, 3:7], axis=1) - 1) > 1e-5)
        or type(original_source_frame) is not int
        or type(target_source_frame) is not int
        or not 0 <= original_source_frame < len(roots)
        or not 0 <= target_source_frame < len(roots)
        or re.fullmatch(r"sha256:[0-9a-f]{64}", source_motion_artifact_hash) is None
    ):
        raise ValueError("bounded finite XYZ/WXYZ source frames and explicit provenance required")
    a, b = roots[original_source_frame], roots[target_source_frame]
    yaw = Rotation.from_quat(np.stack((a, b))[:, [4, 5, 6, 3]]).as_euler("ZYX")[:, 0]

    def matrix(angle: float) -> np.ndarray[Any, Any]:
        return np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])

    offset = np.asarray(prior.ball_offset_xy_m)
    if original_source_frame == target_source_frame:
        new = prior.ball_offset_xy_m
    else:
        point = a[:2] + matrix(yaw[0]) @ offset
        value = matrix(yaw[1]).T @ (point - b[:2])
        if not np.isfinite(value).all() or max(abs(value)) > 1:
            raise ValueError("transported point exceeds original one-meter affordance bound")
        new = (float(value[0]), float(value[1]))
    xy = matrix(yaw[0]).T @ (b[:2] - a[:2])
    delta = float(np.arctan2(np.sin(yaw[1] - yaw[0]), np.cos(yaw[1] - yaw[0])))
    return EntryFrameTransport(
        prior.contract_hash,
        source_motion_artifact_hash,
        hash_bytes(roots.tobytes()),
        original_source_frame,
        target_source_frame,
        prior.ball_offset_xy_m,
        new,
        (float(xy[0]), float(xy[1]), float(b[2] - a[2])),
        delta,
    )
