"""Truthful history frames over private fixture worlds; no trained G1 evidence."""

import mujoco
import numpy as np
import pytest
from test_shared_velocity_forecast import setup

from rosclaw_soccer.rsi.shared_episode_clock import absolute_control_frame
from rosclaw_soccer.rsi.shared_kick_history import measured_kick_sample
from rosclaw_soccer.rsi.shared_native_motor_world import SharedNativeMotorWorld


def copied_completed_fixture():
    actual, _, _ = setup()
    target = {"a/": np.zeros(29)}
    for _ in range(6):
        actual.advance(target)
    scratch = mujoco.MjData(actual.model)
    mujoco.mj_copyData(scratch, actual.model, actual.data)
    owned = SharedNativeMotorWorld(
        actual.model, scratch, players=actual.parameters, ball_body_name="ball"
    )
    return actual, owned, target


def test_new_consumed_counters_stay_local_but_completed_history_is_absolute():
    actual, owned, target = copied_completed_fixture()
    assert owned.control_steps == owned.native_steps == 0
    assert absolute_control_frame(owned, control_origin=6) == 6
    with pytest.raises(ValueError, match="committed"):
        measured_kick_sample(owned, "a/", control_origin=6)
    predicted = owned.advance(target)
    assert absolute_control_frame(owned, control_origin=6) == 7
    sample = measured_kick_sample(owned, "a/", control_origin=6)
    assert sample.frame == 7 and sample.previous_target_frame == 6
    assert owned.control_steps == 1 and owned.native_steps == 10
    assert actual.control_steps == 6 and actual.native_steps == 60
    executed = actual.advance(target)
    for k in ("qpos", "qvel", "torque", "ball_forces"):
        np.testing.assert_array_equal(predicted[k], executed[k])
    assert sample == measured_kick_sample(actual, "a/")


@pytest.mark.parametrize("origin", [True, -1, 361, 5, 7, 6.0])
def test_no_clock_fabrication_to_fit_some_other_history(origin):
    _, owned, _ = copied_completed_fixture()
    before = owned.data.qpos.copy(), owned.data.time, owned.control_steps, owned.native_steps
    with pytest.raises(ValueError):
        absolute_control_frame(owned, control_origin=origin)
    np.testing.assert_array_equal(before[0], owned.data.qpos)
    assert before[1:] == (owned.data.time, owned.control_steps, owned.native_steps)


@pytest.mark.parametrize("origin", [5, 4])
def test_actual_measured_history_can_seed_new_kick_only_at_truthful_absolute_boundary(origin):
    from test_shared_kick_history import MeasuredPolicy, entry

    from rosclaw_soccer.rsi.shared_robonaldo_player import SharedRoboNaldoPlayer

    actual, _, source_history = entry()
    scratch = mujoco.MjData(actual.model)
    mujoco.mj_copyData(scratch, actual.model, actual.data)
    owned = SharedNativeMotorWorld(
        actual.model, scratch, players=actual.parameters, ball_body_name="ball"
    )
    owned.advance({p: np.zeros(29) for p in owned.layouts})
    own_sample = measured_kick_sample(owned, "a/", control_origin=5)
    history = (*source_history[1:], own_sample)
    policy = MeasuredPolicy(own_sample)
    player = SharedRoboNaldoPlayer(owned, "a/", policy, goal=np.array([5.0, 0.0, 0.11]))
    player._forecast_control_origin = origin
    before = owned.data.qpos.copy(), owned.data.qvel.copy(), float(owned.data.time)
    if origin == 5:
        receipt = player.install_measured_entry(history, reference_frame=3)
        assert receipt.startswith("sha256:")
        assert player.measured_entry_receipt == receipt
    else:
        with pytest.raises(ValueError, match="uninterrupted native episode time"):
            player.install_measured_entry(history, reference_frame=3)
        assert player.failed
    np.testing.assert_array_equal(before[0], owned.data.qpos)
    np.testing.assert_array_equal(before[1], owned.data.qvel)
    assert before[2] == owned.data.time
    assert owned.control_steps == 1 and owned.native_steps == 10
    assert actual.control_steps == 5 and actual.native_steps == 50
