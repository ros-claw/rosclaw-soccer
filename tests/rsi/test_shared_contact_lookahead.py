import numpy as np
import pytest
from test_shared_native_motor_world import world

from rosclaw_soccer.rsi.shared_contact_lookahead import lookahead_candidates


def test_two_player_native_lookahead_does_not_modify_measured_state():
    original = world()
    before = original.data.qpos.copy(), original.data.qvel.copy(), original.data.time
    targets = {p: np.zeros(29) for p in original.layouts}
    corrections = np.zeros((2, 29))
    corrections[1, 0] = 0.05
    rows = lookahead_candidates(
        original.model,
        original.data,
        players=original.parameters,
        ball_body_name="ball",
        targets=targets,
        receiver="b/",
        corrections=corrections,
        horizon_controls=3,
    )
    np.testing.assert_array_equal(original.data.qpos, before[0])
    np.testing.assert_array_equal(original.data.qvel, before[1])
    assert original.data.time == before[2]
    assert len(rows) == 2 and all(r["native_steps"] == 30 for r in rows)
    assert all(not r["physically_qualified"] and r["neural_calls"] == 0 for r in rows)


def test_lookahead_rejects_upper_body_corrections():
    original = world()
    correction = np.zeros((1, 29))
    correction[0, 12] = 0.01
    with pytest.raises(ValueError):
        lookahead_candidates(
            original.model,
            original.data,
            players=original.parameters,
            ball_body_name="ball",
            targets={p: np.zeros(29) for p in original.layouts},
            receiver="b/",
            corrections=correction,
            horizon_controls=3,
        )
