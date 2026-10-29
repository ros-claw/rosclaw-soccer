from types import SimpleNamespace

import numpy as np
import pytest

from rosclaw_soccer.rsi.causal_ball_reference import CausalBallReference


def policy():
    return SimpleNamespace(
        reference_source="NPZmotion",
        counter_step=0,
        ref_right_ankle_future_pos=np.zeros((20, 3), dtype=np.float32),
        ref_object_pos=np.zeros((20, 3), dtype=np.float32),
        ref_torso_future_pos=np.ones((20, 3), dtype=np.float32),
    )


def test_measured_feedback_is_bounded_slewed_and_nonaccumulating():
    p = policy()
    module = CausalBallReference(p)
    ball = np.array([1.0, 0.0, 0.11])
    previous = np.zeros(2)
    torso = p.ref_torso_future_pos.copy()
    for tick in range(20):
        p.counter_step = tick
        offset = module.apply(ball)
        assert np.linalg.norm(offset) <= 0.35 + 1e-12
        assert np.linalg.norm(offset - previous) <= 0.02 + 1e-12
        assert np.allclose(p.ref_right_ankle_future_pos[:, 0], offset[0])
        assert np.array_equal(p.ref_right_ankle_future_pos[:, 2], np.zeros(20))
        previous = offset
    assert previous[0] == pytest.approx(0.35)
    assert np.array_equal(p.ref_torso_future_pos, torso)
    assert np.array_equal(p.ref_object_pos, np.zeros((20, 3)))
    p.counter_step = 20
    assert module.apply(ball)[0] == pytest.approx(0.33)


def test_sequential_fresh_ball_and_private_reference_required():
    p = policy()
    module = CausalBallReference(p)
    with pytest.raises(ValueError):
        module.apply(np.array([np.nan, 0.0, 0.0]))
    module.apply(np.array([0.1, 0.0, 0.11]))
    with pytest.raises(ValueError):
        module.apply(np.array([0.1, 0.0, 0.11]))
    p.counter_step = 1
    p.ref_object_pos[0, 0] = 0.1
    with pytest.raises(ValueError):
        module.apply(np.array([0.1, 0.0, 0.11]))
