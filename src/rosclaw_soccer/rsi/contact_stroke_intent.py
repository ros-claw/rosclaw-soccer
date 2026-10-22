"""Coherent impact/retraction joint proposal for a bounded SIM teacher search.

This is NOT a learned policy, qualified skill, or command executor. The caller
must independently bound the TOTAL action relative to the frozen nominal,
advance real simulation, and retain every failed physical search response.
"""

import numpy as np

from rosclaw_soccer.providers.g1.joint_contract import G1_DDS_JOINT_NAMES

LEG_SUFFIXES = ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")


def contact_stroke_delta(*, foot: str, frame: int, center: int, latent: np.ndarray) -> np.ndarray:
    """Two smooth overlapping pulses, twelve teacher-search variables in HG29.

    The second pulse begins BEFORE impact so a planned release/braking action
    need not wait for the next20ms observation after an already-late contact.
    Center is privileged TRAIN planning data, never an asserted actor input.
    """
    if (
        foot not in ("left", "right")
        or type(frame) is not int
        or not 0 <= frame < 300
        or type(center) is not int
        or not 98 <= center <= 180
        or not isinstance(latent, np.ndarray)
        or latent.shape != (2, 6)
        or latent.dtype.kind != "f"
        or not np.isfinite(latent).all()
        or (abs(latent) > 1).any()
    ):
        raise ValueError("bounded explicit TRAIN contact stroke proposal required")
    phase = (frame - np.array([center - 6, center + 2])) / np.array([12.0, 8.0])
    weights = np.where(abs(phase) < 1, (1 - phase**2) ** 3, 0.0)
    leg = 0.2 * np.tanh((weights[:, None] * latent).sum(axis=0))
    delta = np.zeros(29, dtype=np.float64)
    indices = [G1_DDS_JOINT_NAMES.index(f"{foot}_{s}_joint") for s in LEG_SUFFIXES]
    delta[indices] = leg
    return delta
