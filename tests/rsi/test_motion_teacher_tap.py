from types import SimpleNamespace

import numpy as np
import pytest

from scripts.rsi_motion_teacher_tap import MotionTeacherTap


class Session:
    def __init__(self, out):
        self.out = out

    def run(self, _, inputs):
        return [self.out]


def test_teacher_capture_preserves_outputs_and_binds_consumed_action_history():
    token = np.arange(-16, 16, dtype=np.float32).repeat(2)[None] / 16
    action = np.zeros((1, 29), dtype=np.float32)
    backend = SimpleNamespace(_encoder=Session(token), _decoder=Session(action))
    tap = MotionTeacherTap(backend)
    x = np.zeros((1, 1247), dtype=np.float32)
    y = np.zeros((1, 994), dtype=np.float32)
    y[0, :64] = token
    assert backend._encoder.run(None, {"x": x})[0] is token
    assert backend._decoder.run(None, {"x": y})[0] is action
    label = tap.label(np.zeros(930, dtype=np.float32))
    np.testing.assert_array_equal(label["teacher_motion_token"], token[0])
    with pytest.raises(ValueError):
        tap.label(np.zeros(930, dtype=np.float32))


def test_teacher_history_mismatch_is_rejected_not_silently_replaced():
    backend = SimpleNamespace(
        _encoder=Session(np.zeros((1, 64), dtype=np.float32)),
        _decoder=Session(np.zeros((1, 29), dtype=np.float32)),
    )
    tap = MotionTeacherTap(backend)
    backend._encoder.run(None, {"x": np.zeros((1, 1247), dtype=np.float32)})
    backend._decoder.run(None, {"x": np.zeros((1, 994), dtype=np.float32)})
    with pytest.raises((ValueError, AssertionError)):
        tap.label(np.ones(930, dtype=np.float32))
