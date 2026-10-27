import numpy as np
import pytest

from rosclaw_soccer.rsi.relay_fall_samples import RelayLayout, relay_fall_samples


def fixture(fail_b=False, fail_a=False):
    q0, v0 = np.zeros(79), np.zeros(76)
    q0[3], q0[10] = 1, 1
    q = np.repeat(q0[None, None, :], 50, axis=0).reshape(5, 10, 79)
    v = np.zeros((5, 10, 76))
    target = np.arange(5 * 29, dtype=np.float64).reshape(5, 29)
    failed = np.zeros((5, 2), dtype=np.bool_)
    failed[-1] = [fail_a, fail_b]
    layout = RelayLayout(0, 0, 7, 6, tuple(range(14, 43)), tuple(range(12, 41)))
    return dict(
        initial_qpos=q0,
        initial_qvel=v0,
        qpos=q,
        qvel=v,
        target_b=target,
        body_failed=failed,
        layout=layout,
        horizon_controls=3,
    )


def test_fall_uses_only_pre_action_states_and_previous_executed_target():
    data = fixture(fail_b=True)
    x, y, t = relay_fall_samples(**data)
    assert x.shape == (5, 104)
    assert y.tolist() == [False, False, True, True, True]
    assert t.tolist() == list(range(5))
    assert np.array_equal(x[2, -30:-1], data["target_b"][1])
    assert x[0, -1] == 0 and x[1, -1] == 1
    changed = fixture(fail_b=True)
    changed["target_b"][2:] = -100
    assert np.array_equal(relay_fall_samples(**changed)[0][:3], x[:3])


def test_surviving_or_other_player_failure_tail_is_censored():
    for data in (fixture(), fixture(fail_a=True)):
        x, y, t = relay_fall_samples(**data)
        assert x.shape == (3, 104)
        assert not y.any()
        assert t.tolist() == [0, 1, 2]


def test_invalid_or_mid_episode_failure_rejected():
    data = fixture()
    data["body_failed"][1, 1] = True
    with pytest.raises(ValueError):
        relay_fall_samples(**data)
