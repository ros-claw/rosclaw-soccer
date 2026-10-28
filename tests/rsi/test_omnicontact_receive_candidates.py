import numpy as np
import pytest

from rosclaw_soccer.rsi.omnicontact_receive_candidates import candidate_events


def test_screen_identifies_bounded_reference_event_not_physical_success():
    n, fps, frame = 100, 100, 50
    ball = np.zeros((n, 3))
    ball[:frame, 0] = -0.1
    velocity = np.zeros((n, 3))
    velocity[:frame, 0] = 1
    body = np.zeros((n, 39, 3))
    contact = np.zeros((n, 4, 1), dtype=np.uint8)
    contact[frame : frame + 4, 0, 0] = 1
    rows = candidate_events(ball, velocity, body, contact, fps)
    assert len(rows) == 1 and rows[0]["frame"] == frame and rows[0]["foot"] == "left"
    assert rows[0]["reference_only_not_actual_contact"] is True
    velocity[frame:, 0] = 1
    assert candidate_events(ball, velocity, body, contact, fps) == []


def test_nonbinary_contact_is_rejected():
    n = 100
    contact = np.zeros((n, 4, 1))
    contact[50, 0, 0] = 0.5
    with pytest.raises(ValueError, match="binary"):
        candidate_events(np.zeros((n, 3)), np.zeros((n, 3)), np.zeros((n, 39, 3)), contact, 100)
