from types import SimpleNamespace

import numpy as np
import pytest

from rosclaw_soccer.rsi.paid_balance_recovery import (
    BalanceRecovery,
    body_feedback,
    feedback_delta,
    qualified_reference_prefix,
    transfer_reference_prefix,
)


def state():
    q, v = np.zeros(43), np.zeros(41)
    q[2], q[3] = 0.78, 1
    return q, v


def test_stationary_upright_feedback_and_zero_gain_have_zero_request():
    q, v = state()
    feature = body_feedback(q, v)
    assert np.array_equal(feature, np.zeros(8))
    assert np.array_equal(feedback_delta(feature, np.ones((9, 4))), np.zeros(29))
    v[:3] = [0.2, -0.3, 0.1]
    assert np.array_equal(feedback_delta(body_feedback(q, v), np.zeros((9, 4))), np.zeros(29))


def test_world_translation_is_rotated_but_freejoint_rotation_is_already_local():
    q, v = state()
    q[3:7] = [np.sqrt(0.5), 0, 0, np.sqrt(0.5)]
    v[:6] = [0, 1, 0, 0.2, -0.4, 0]
    feature = body_feedback(q, v)
    assert np.allclose(feature[:4], [0, 0, 0.2, -0.4])
    assert np.allclose(feature[4:6], [1, 0])


def test_gains_are_bounded_named_joint_outputs_not_a_body_pose_write():
    request = feedback_delta(np.full(8, 100.0), np.ones((9, 4)))
    assert request.shape == (29,) and np.abs(request).max() <= 0.2
    # No hip-yaw/waist-yaw or elbow/wrist authority in this teacher.
    assert np.array_equal(request[[2, 8, 12, 18, 19, 20, 21, 25, 26, 27, 28]], np.zeros(11))


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_nonfinite_feedback_rejected_without_repair(bad):
    q, v = state()
    q[2] = bad
    with pytest.raises(ValueError):
        body_feedback(q, v)


def test_nonunit_quaternion_not_silently_normalized():
    q, v = state()
    q[3] = 2
    with pytest.raises(ValueError, match="unit"):
        body_feedback(q, v)


def test_qualified_prefix_contains_all_actual_nn_rows_but_not_a_padded_hold():
    raw = dict(
        NN_action=np.ones((2, 29)),
        target=np.arange(62 * 29).reshape(62, 29),
        neural_control=np.r_[np.ones(2, bool), np.zeros(60, bool)],
        body_failed=np.zeros(62, bool),
        initial_time=np.array(0.0),
    )
    prefix = qualified_reference_prefix(raw)
    assert len(prefix["target"]) == 2
    assert np.array_equal(prefix["NN_action"], raw["NN_action"])
    assert np.array_equal(prefix["target"], raw["target"][:2])
    prefix["NN_action"][0, 0] = 9
    assert raw["NN_action"][0, 0] == 1
    raw["body_failed"][-1] = True
    with pytest.raises(ValueError, match="healthy source"):
        qualified_reference_prefix(raw)


def test_recovery_anchor_slew_budget_and_actual_action(monkeypatch):
    import mujoco

    monkeypatch.setattr(mujoco, "mj_forward", lambda *_: None)
    ep = SimpleNamespace(
        done=True,
        termination_reason="REFERENCE_COMPLETE",
        target=np.zeros(29),
        last_action=np.zeros(29, np.float32),
        offset=np.zeros(29),
        scale=np.ones(29),
        order=np.arange(29),
        env=SimpleNamespace(model=None, data=None),
        _integrate_target=lambda neural: dict(body_failed=False, neural_control=neural),
    )
    recovery = BalanceRecovery(ep)
    for i in range(100):
        row = recovery.step(np.full(29, 0.2))
        assert row["neural_control"] is False
        assert np.abs(ep.target).max() <= 0.2 + 1e-6
        if i == 0:
            assert np.allclose(ep.target, 0.02)
        assert np.array_equal(ep.target, ep.offset + ep.last_action[ep.order] * ep.scale)
    with pytest.raises(ValueError, match="budget"):
        recovery.step(np.zeros(29))
    ep.termination_reason = "BODY_FAILURE"
    with pytest.raises(ValueError, match="healthy"):
        BalanceRecovery(ep)


def test_transfer_keeps_actual_early_body_failure_without_padding_or_dropping():
    raw = dict(
        NN_action=np.ones((5, 29)),
        target=np.ones((5, 29)),
        neural_control=np.ones(5, bool),
        body_failed=np.array([False] * 4 + [True]),
        initial_time=np.array(0.0),
    )
    result = transfer_reference_prefix(raw, horizon=257)
    assert len(result["target"]) == 5 and result["body_failed"][-1]
    assert set(result) == set(raw)
    raw["body_failed"][-1] = False
    with pytest.raises(ValueError, match="actual body-failure"):
        transfer_reference_prefix(raw, horizon=257)


def test_transfer_can_measure_prefix_before_an_existing_tail_failure():
    raw = dict(
        NN_action=np.ones((3, 29)),
        target=np.ones((8, 29)),
        neural_control=np.r_[np.ones(3, bool), np.zeros(5, bool)],
        body_failed=np.array([False] * 7 + [True]),
        initial_time=np.array(0.0),
    )
    prefix = transfer_reference_prefix(raw, horizon=3)
    assert len(prefix["target"]) == 3 and not prefix["body_failed"].any()
