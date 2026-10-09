"""Before-action task history for offline critics, never actuator authority.

Fixed300 contact-course semantics, matching temporal_contact_rewards. This is
not a generic robot safety certificate or an actor observation contract.
"""

from typing import Any

import numpy as np

FEATURE_NAMES = (
    "remaining_time",
    "previous_contact_seen",
    "elapsed_since_previous_first_contact",
    "previous_dirty_contact_seen",
    "previous_boundary_crossing_seen",
    "previous_height_violation_seen",
)


def before_action_task_state(trace: Any) -> np.ndarray[Any, Any]:
    shapes = {
        "force_n": (300, 1, 6),
        "pelvis_z_per_substep_m": (300, 1, 10),
        "ball_position_after_step_m": (300, 1, 3),
    }
    try:
        arrays = {key: np.asarray(trace[key]) for key in shapes}
    except (KeyError, TypeError, IndexError) as error:
        raise ValueError("complete physical contact-course trace mapping required") from error
    if any(
        value.shape != shapes[key] or value.dtype.kind != "f" or not np.isfinite(value).all()
        for key, value in arrays.items()
    ) or np.any(arrays["force_n"] < 0):
        raise ValueError("complete finite physical contact-course histories required")
    forces = arrays["force_n"][:, 0]
    heights = arrays["pelvis_z_per_substep_m"][:, 0]
    ball = arrays["ball_position_after_step_m"][:, 0]
    state = np.zeros((300, 6), dtype=np.float64)
    first: int | None = None
    dirty = outside = unsafe = False
    for frame in range(300):
        # Frame f's measured post-action values are unknown before action f.
        if frame:
            previous = frame - 1
            contact = forces[previous] > 1
            if first is None and np.any(contact):
                first = previous
            dirty = dirty or bool(np.any(contact[2:]))
            outside = outside or bool(abs(ball[previous, 1]) > 4)
            unsafe = unsafe or bool(np.min(heights[previous]) < 0.65)
        state[frame] = (
            (300 - frame) / 300,
            first is not None,
            0 if first is None else (frame - first) / 300,
            dirty,
            outside,
            unsafe,
        )
    state.flags.writeable = False
    return state
