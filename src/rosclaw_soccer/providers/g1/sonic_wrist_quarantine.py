"""Explicit reference ablation: hold six wrist channels at measured entry.

This is NOT source repair or full29-joint imitation. Original source hashes and
removed channel variation are retained; legs, waist, shoulders, elbows, root
and time are unchanged. The physical robot's wrists are never set or reset.
"""

from dataclasses import dataclass

import numpy as np

from rosclaw_soccer.providers.g1.joint_contract import G1_DDS_JOINT_NAMES
from rosclaw_soccer.providers.g1.sonic_reference_preparation import _pose_array
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json

WRIST_REFERENCE_NAMES = tuple(n for n in G1_DDS_JOINT_NAMES if "_wrist_" in n)
WRIST_REFERENCE_COLUMNS = tuple(7 + G1_DDS_JOINT_NAMES.index(n) for n in WRIST_REFERENCE_NAMES)


@dataclass(frozen=True)
class WristReferenceQuarantine:
    original_source_hash: str
    quarantined_source_hash: str
    measured_entry_hash: str
    maximum_removed_adjacent_change_rad: tuple[float, ...]

    @property
    def contract_hash(self) -> str:
        return hash_json(
            dict(
                schema="soccer.sonic_reference_wrist_quarantine.v1",
                **self.__dict__,
                held_channels=WRIST_REFERENCE_NAMES,
                hold_value="measured_entry_only",
                original_source_repaired=False,
                source_29joint_imitation_qualified=False,
                body_state_modified=False,
                activation_ceiling="SIM_ONLY",
                hardware_authorized=False,
            )
        )


def quarantine_wrist_reference(
    source_poses: np.ndarray,
    measured_entry_pose: np.ndarray,
) -> tuple[np.ndarray, WristReferenceQuarantine]:
    """Return a private canonical copy and explicit provenance for this ablation.

    Nonfinite channels are rejected even if they would be held. Non-wrist
    source limits and quaternions are checked, never clipped or normalized.
    Source continuity still must pass the separate timed-reference admission.
    """
    entry = _pose_array(measured_entry_pose, rows=False)
    if (
        type(source_poses) is not np.ndarray
        or source_poses.dtype.kind != "f"
        or source_poses.ndim != 2
        or source_poses.shape[1] != 36
        or not 2 <= len(source_poses) <= 10000
        or not np.isfinite(source_poses).all()
    ):
        raise ValueError("complete finite named source required, even on quarantined channels")
    original = np.asarray(source_poses, dtype=np.float64)
    result = original.copy()
    result[:, WRIST_REFERENCE_COLUMNS] = entry[list(WRIST_REFERENCE_COLUMNS)]
    result = _pose_array(result, rows=True)
    variation = np.max(abs(np.diff(original[:, WRIST_REFERENCE_COLUMNS], axis=0)), axis=0)
    receipt = WristReferenceQuarantine(
        hash_bytes(original.tobytes()),
        hash_bytes(result.tobytes()),
        hash_bytes(entry.tobytes()),
        tuple(float(v) for v in variation),
    )
    return result, receipt
