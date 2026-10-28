import numpy as np
import pytest
from test_contact_reference_ik import fixture

from rosclaw_soccer.rsi.reference_leg_reconstruction import (
    reconstruct_right_leg,
    shifted_reference_frames,
)


def test_phase_shift_moves_only_reference_window_and_rejects_overflow():
    original = np.arange(30, 71)
    shifted = shifted_reference_frames(original, -17, 101)
    np.testing.assert_array_equal(shifted, np.arange(13, 54))
    np.testing.assert_array_equal(original, np.arange(30, 71))
    with pytest.raises(ValueError):
        shifted_reference_frames(original, -30, 101)
    with pytest.raises(ValueError):
        shifted_reference_frames(np.arange(3, 8), -3, 101)


def test_reconstruction_rejects_nonzero_window_endpoints_without_modifying_source():
    model, _, layout = fixture(0.12)
    motion = {"joint_pos": np.zeros((8, 29)), "fps": np.array([50.0])}
    offsets = np.zeros((3, 6), dtype=np.float64)
    offsets[0, 0] = 0.1
    with pytest.raises(ValueError):
        reconstruct_right_leg(model, layout, motion, np.array([2, 3, 4]), offsets)
    np.testing.assert_array_equal(motion["joint_pos"], 0)


def test_reconstruction_rejects_unbounded_frame_sequence():
    model, _, layout = fixture(0.12)
    motion = {"joint_pos": np.zeros((8, 29)), "fps": np.array([50.0])}
    with pytest.raises(ValueError):
        reconstruct_right_leg(model, layout, motion, np.array([2, 4, 5]), np.zeros((3, 6)))
