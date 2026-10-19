"""Explicit timed reference preparation, never body-state animation.

The caller supplies externally licensed, named canonical29 poses and its timing
contract. No source dataset is bundled. This only builds a tracker input: root
translation is metadata, not SONIC Cartesian tracking authority. Neither entry
interpolation nor a source hash establishes dynamic feasibility or goal labels.
"""

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation, Slerp  # type: ignore[import-untyped]

from rosclaw_soccer.providers.g1.sonic_pose_reference import SonicPoseReference
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json


@dataclass(frozen=True)
class PreparedSonicReference:
    reference: SonicPoseReference
    source_pose_hash: str
    source_frequency_hz: float
    source_frames: int
    sampled_frames: int
    entry_intervals: int
    source_duration_sec: float
    sampled_duration_sec: float
    root_translation_shift_m: tuple[float, float, float]
    heading_rotation_rad: float
    source_quaternion_norm_error: float

    @property
    def contract_hash(self) -> str:
        return hash_json(
            {
                "schema": "soccer.sonic_timed_reference_preparation.v1",
                **self.__dict__,
                "reference": self.reference.contract_hash,
                "time_stretch": False,
                "entry": "explicit_quintic_position_slerp_orientation",
                "root_alignment": "yaw_and_translation_from_first_source_to_measured_entry",
                "root_translation_is_tracker_authority": False,
                "modifies_physical_body_state": False,
                "dynamic_feasibility_qualified": False,
                "hardware_authorized": False,
                "activation_ceiling": "SIM_ONLY",
            }
        )


def _pose_array(value: np.ndarray, *, rows: bool) -> np.ndarray:
    if (
        type(value) is not np.ndarray
        or value.dtype.kind != "f"
        or (value.ndim != 2 or value.shape[1] != 36 if rows else value.shape != (36,))
        or not np.isfinite(value).all()
    ):
        raise ValueError("finite named canonical36 floating pose array required")
    array = np.asarray(value, dtype=np.float64).copy()
    matrix = array if rows else array[None]
    if np.any(np.abs(np.linalg.norm(matrix[:, 3:7], axis=1) - 1) > 1e-5) or np.any(
        np.abs(matrix[:, 7:]) > 3.2
    ):
        raise ValueError("unit wxyz quaternion and bounded29 joints required; no clipping")
    return array


def prepare_sonic_reference(
    source_poses: np.ndarray,
    measured_entry_pose: np.ndarray,
    *,
    source_frequency_hz: float,
    entry_intervals: int,
    source_evidence_hash: str,
) -> PreparedSonicReference:
    """Resample to50Hz and explicitly bridge from the measured, unmoved body.

    Source timestamps are i/fps, not inferred from motion. Sampling never extends
    past the last source timestamp; at most one50Hz interval is unrepresented.
    All29 joint channels are retained. RootXYZ is rigidly re-anchored as reference
    metadata, not calibrated to the current body's floor or foot geometry.
    """
    source = _pose_array(source_poses, rows=True)
    entry = _pose_array(measured_entry_pose, rows=False)
    if (
        type(source_frequency_hz) not in (int, float)
        or not np.isfinite(source_frequency_hz)
        or not 20 <= source_frequency_hz <= 240
        or type(entry_intervals) is not int
        or not 10 <= entry_intervals <= 150
        or not 10 <= len(source) <= 10000
    ):
        raise ValueError("explicit20..240Hz source and10..150 interval entry required")
    duration = (len(source) - 1) / source_frequency_hz
    count = int(np.floor(duration * 50)) + 1
    if count < 10 or count + entry_intervals > 1001:
        raise ValueError("bounded complete source and entry must fit reference horizon")
    times = np.arange(len(source), dtype=np.float64) / source_frequency_hz
    sampled_times = np.arange(count, dtype=np.float64) / 50
    sampled = np.empty((count, 36), dtype=np.float64)
    for col in (*range(3), *range(7, 36)):
        sampled[:, col] = np.interp(sampled_times, times, source[:, col])
    rotations = Rotation.from_quat(source[:, [4, 5, 6, 3]])
    reference_rotations = Slerp(times, rotations)(sampled_times)
    first_yaw = float(rotations[0].as_euler("ZYX")[0])
    entry_yaw = float(Rotation.from_quat(entry[[4, 5, 6, 3]]).as_euler("ZYX")[0])
    yaw_delta = float(np.arctan2(np.sin(entry_yaw - first_yaw), np.cos(entry_yaw - first_yaw)))
    yaw = Rotation.from_rotvec([0, 0, yaw_delta])
    shift = entry[:3] - yaw.apply(source[0, :3])
    sampled[:, :3] = yaw.apply(sampled[:, :3]) + shift
    sampled[:, 3:7] = (yaw * reference_rotations).as_quat()[:, [3, 0, 1, 2]]
    # Smooth reference entry is NOT a physical state reset. The tracker must
    # still react to its own measured proprioception throughout every step.
    u = np.arange(entry_intervals + 1, dtype=np.float64) / entry_intervals
    blend = u**3 * (10 + u * (-15 + 6 * u))
    bridge = entry + blend[:, None] * (sampled[0] - entry)
    bridge[:, 3:7] = Slerp(
        [0, 1], Rotation.from_quat(np.stack((entry, sampled[0]))[:, [4, 5, 6, 3]])
    )(blend).as_quat()[:, [3, 0, 1, 2]]
    prepared = np.concatenate((bridge[:-1], sampled))
    # Preserve exact measured entry bits; no implicit normalization/teleport.
    prepared[0] = entry
    reference = SonicPoseReference(
        tuple(tuple(float(v) for v in row) for row in prepared), source_evidence_hash
    )
    return PreparedSonicReference(
        reference=reference,
        source_pose_hash=hash_bytes(source.tobytes()),
        source_frequency_hz=float(source_frequency_hz),
        source_frames=len(source),
        sampled_frames=count,
        entry_intervals=entry_intervals,
        source_duration_sec=float(duration),
        sampled_duration_sec=(count - 1) / 50,
        root_translation_shift_m=(float(shift[0]), float(shift[1]), float(shift[2])),
        heading_rotation_rad=yaw_delta,
        source_quaternion_norm_error=float(
            np.max(np.abs(np.linalg.norm(source[:, 3:7], axis=1) - 1))
        ),
    )
