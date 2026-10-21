"""Explicit SIM feed-forward policy history restoration, not weight adaptation."""

from typing import Any

import numpy as np

from rosclaw_soccer.providers.g1.velocity_policy import phase_delta


def restore_native_policy_history(
    session: Any, *, frame: int, previous_action: np.ndarray, phase_before: float
) -> None:
    if (
        type(frame) is not int
        or not 0 <= frame < 300
        or not isinstance(previous_action, np.ndarray)
        or previous_action.shape != (29,)
        or previous_action.dtype != np.float32
        or not np.isfinite(previous_action).all()
        or type(phase_before) is not float
        or not np.isfinite(phase_before)
    ):
        raise ValueError("exact finite recorded policy history required")
    phase = phase_delta()
    for _ in range(frame):
        phase = np.float32(np.fmod(np.float32(phase + phase_delta()), np.float32(1)))
    if float(phase) != phase_before:
        raise ValueError("recorded gait phase is not aligned to reference frame")
    if (
        session.next_frame != 0
        or session.neural_calls != 0
        or session.attempted_neural_calls != 0
        or session.faulted
        or session.reference_mode != "declared79"
    ):
        raise ValueError("only fresh explicit177 SIM sessions may restore history")
    if frame == 0 and np.count_nonzero(previous_action):
        raise ValueError("frame-zero history must match original zero action")
    session.last_action = previous_action.copy()
    session.phase = np.float32(phase_before)
    session.next_frame = frame
