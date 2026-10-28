from types import SimpleNamespace

import numpy as np
from test_shared_native_motor_world import world

from rosclaw_soccer.rsi.shared_contact_neural_lookahead import neural_lookahead


class ZeroActor:
    def run(self, names, feed):
        assert names == ["actions"] and feed["obs"].shape == (1, 98)
        return [np.zeros((1, 29), np.float32)]


def test_branch_reinfers_and_preserves_initial_snapshot_and_policy_history():
    w = world()
    before_q, before_v, before_t = w.data.qpos.copy(), w.data.qvel.copy(), w.data.time
    memory = np.zeros(29, np.float32)
    policy = SimpleNamespace(
        params=SimpleNamespace(
            defaults=(0.0,) * 29, action_scale=(1.0,) * 29, action_offset=(0.0,) * 29
        ),
        session=ZeroActor(),
    )
    rows = neural_lookahead(
        w.model,
        w.data,
        players=w.parameters,
        ball_body_name="ball",
        receiver="b/",
        passer_target=np.zeros(29),
        policy=policy,
        last_action=memory,
        phase=np.float32(0),
        command=np.zeros(3, np.float32),
        corrections=np.zeros((2, 29)),
        horizon_controls=2,
    )
    np.testing.assert_array_equal(w.data.qpos, before_q)
    np.testing.assert_array_equal(w.data.qvel, before_v)
    np.testing.assert_array_equal(memory, 0)
    assert w.data.time == before_t
    assert all(r["native_steps"] == 20 and r["neural_calls"] == 2 for r in rows)
    assert all(not r["passer_neural_resume"] and not r["physically_qualified"] for r in rows)
