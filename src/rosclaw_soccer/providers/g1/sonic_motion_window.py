"""Pre-physics selection of a bounded dynamic leg-motion reference window.

This selection uses source kinematics only, never ball outcomes, simulator
rollouts or retention scores. A motion category is not a task success label.
The whole source is not repaired or certified by extracting a valid window.
"""

from dataclasses import dataclass

import numpy as np

from rosclaw_soccer.providers.g1.sonic_reference_preparation import (
    PreparedSonicReference,
    _pose_array,
    prepare_sonic_reference,
)


@dataclass(frozen=True)
class SonicMotionWindow:
    source_start_frame: int
    source_stop_frame_exclusive: int
    source_peak_leg_interval: int
    source_peak_leg_speed_rad_per_sec: float
    prepared: PreparedSonicReference


def select_dynamic_leg_window(
    source_poses: np.ndarray,
    measured_entry_pose: np.ndarray,
    *,
    source_evidence_hash: str,
) -> SonicMotionWindow:
    """Fixed120Hz,4s window centered on first maximum12-leg-joint speed.

    All29 joint channels, root orientation and continuous timing are retained.
    Geometry admission is unchanged: an invalid selected window is rejected,
    not moved to a less difficult part of the same clip or silently smoothed.
    """
    source = _pose_array(source_poses, rows=True)
    if not 481 <= len(source) <= 10000:
        raise ValueError("complete120Hz source with at least4seconds required")
    leg_speed = np.linalg.norm(np.diff(source[:, 7:19], axis=0), axis=1) * 120
    peak = int(np.argmax(leg_speed))
    start = min(max(peak - 240, 0), len(source) - 481)
    prepared = prepare_sonic_reference(
        source[start : start + 481],
        measured_entry_pose,
        source_frequency_hz=120,
        entry_intervals=50,
        source_evidence_hash=source_evidence_hash,
    )
    return SonicMotionWindow(start, start + 481, peak, float(leg_speed[peak]), prepared)
