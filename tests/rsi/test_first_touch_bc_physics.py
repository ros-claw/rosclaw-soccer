import numpy as np
import pytest

from scripts.rsi_first_touch_bc_feasibility import causal_examples
from scripts.rsi_first_touch_bc_physics import phase_entry, student_observation


def test_live_canonical_observation_matches_offline_sample():
    q = np.arange(20 * 43, dtype=float).reshape(20, 43)
    v = np.arange(20 * 41, dtype=float).reshape(20, 41)
    targets = np.arange(58, dtype=float).reshape(2, 29)
    offline, _ = causal_examples(dict(qpos=q, qvel=v, joint_targets=targets))
    q_can = np.r_[q[9, 7:43], q[9, :7]]
    v_can = np.r_[v[9, 6:41], v[9, :6]]
    np.testing.assert_array_equal(student_observation(q_can, v_can, targets[0]), offline[0])


def test_live_observation_rejects_invalid_state():
    with pytest.raises(ValueError):
        student_observation(np.zeros(42), np.zeros(41), np.zeros(29))
    q = np.zeros(43)
    q[1] = float("nan")
    # Global root x/y are omitted unless needed for relative ball position.
    with pytest.raises(ValueError):
        student_observation(q, np.zeros(41), np.zeros(29))


def test_kick_phase_entry_requires_causal_bounded_state():
    assert phase_entry(50, [0.9, 0.25, 0.0])
    assert not phase_entry(49, [0.9, 0.25, 0.0])
    assert not phase_entry(240, [0.9, 0.25, 0.0])
    assert not phase_entry(50, [0.0, 0.0, 0.0])
    assert not phase_entry(50, [0.9, 0.251, 0.0])
    with pytest.raises(ValueError):
        phase_entry(50, [float("nan"), 0.0, 0.0])
