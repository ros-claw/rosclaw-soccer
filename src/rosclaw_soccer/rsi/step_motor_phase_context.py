"""Causal contact-phase memory for independently protected motor learning heads."""

import numpy as np


class ContactPhaseMemory:
    """Only previous completed contact forces may latch a physical event."""

    def __init__(self) -> None:
        self.last_frame = -1
        self.first_contact_frame: int | None = None

    def advance(self, frame: int, previous_contact_forces: np.ndarray) -> int:
        forces = np.asarray(previous_contact_forces, dtype=np.float64)
        if (
            type(frame) is not int
            or frame != self.last_frame + 1
            or frame >= 3000
            or forces.shape != (6,)
            or not np.isfinite(forces).all()
            or np.any(forces < 0)
            or (frame == 0 and np.any(forces != 0))
        ):
            raise ValueError("sequential causal completed-force observation required")
        if frame and self.first_contact_frame is None and np.any(forces > 1.0):
            self.first_contact_frame = frame - 1
        self.last_frame = frame
        if self.first_contact_frame is None:
            return 0
        return 1 if frame - self.first_contact_frame <= 20 else 2


def phase_sequence(forces: np.ndarray) -> np.ndarray:
    values = np.asarray(forces, dtype=np.float64)
    if values.shape != (300, 6) or not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("complete finite actual contact trace required")
    memory = ContactPhaseMemory()
    return np.asarray(
        [memory.advance(f, values[f - 1] if f else np.zeros(6)) for f in range(300)], dtype=np.int64
    )
