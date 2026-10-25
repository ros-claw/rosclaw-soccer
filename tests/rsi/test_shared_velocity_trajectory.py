"""Fixture MuJoCo + synthetic NN only, not trained G1 trajectory evidence."""

import numpy as np
import pytest
from test_shared_velocity_forecast import setup

from rosclaw_soccer.rsi.shared_target_forecast import compiled_model_hash
from rosclaw_soccer.rsi.shared_velocity_trajectory import SharedVelocityTrajectoryForecaster
from rosclaw_soccer.rsi.walk_kick_approach_command import approach_command


def test_future_recomputes_own_commands_nn_and_native_body_without_donor_mutation():
    world, stream, _ = setup()
    forecaster = SharedVelocityTrajectoryForecaster(
        world, stream, expected_model_hash=compiled_model_hash(world.model), native_step_budget=160
    )
    params = dict(speed_mps=0.8, entry_gap_m=0.9, root_lateral_offset_m=0.0)
    previous = np.zeros(3, np.float32)
    prediction = forecaster.forecast(controls=16, previous_command=previous, parameters=params)
    assert world.native_steps == stream.neural_calls == 0
    assert prediction["copied_next_frame"] == 16
    assert prediction["controls"] == prediction["requested_controls"] == 16
    assert prediction["native_substep_exception_raw_complete"]
    lo = world.layouts["a/"]
    for frame in range(16):
        q, v = lo.canonical(world.data)
        command = approach_command(
            q=q,
            v=v,
            rotation=world.data.xmat[lo.pelvis].reshape(3, 3),
            cached_ball_position=world.data.xpos[lo.ball],
            previous=previous,
            parameters=params,
            fast=False,
            feedback=False,
        )
        target = stream.propose(q, v, lo.angular_velocity_body(world.data), command, frame=frame)
        np.testing.assert_array_equal(command, prediction["commands"][frame])
        np.testing.assert_array_equal(target, prediction["joint_targets"][frame])
        np.testing.assert_array_equal(stream.last_observation, prediction["observations"][frame])
        actual = world.advance({"a/": target})
        for k in ("qpos", "qvel", "torque", "ball_forces"):
            np.testing.assert_array_equal(actual[k], prediction[k][frame * 10 : (frame + 1) * 10])
        previous = command.copy()
    assert forecaster.counters()["future_velocity_nn_calls"] == 16
    with pytest.raises(RuntimeError, match="exhausted"):
        forecaster.forecast(controls=1, previous_command=previous, parameters=params)


@pytest.mark.parametrize("controls", [0, 33, True, 1.0])
def test_no_implicit_horizon_expansion(controls):
    world, stream, _ = setup()
    forecaster = SharedVelocityTrajectoryForecaster(
        world, stream, expected_model_hash=compiled_model_hash(world.model), native_step_budget=320
    )
    with pytest.raises(ValueError):
        forecaster.forecast(
            controls=controls, previous_command=np.zeros(3, np.float32), parameters={}
        )
    assert forecaster.counters()["attempted_native_steps"] == 0
    assert stream.neural_calls == world.native_steps == 0


def test_body_failure_retains_completed_raw_and_charges_full_requested_budget():
    world, stream, _ = setup()
    world.model.opt.gravity[:] = [0, 0, -2000]
    forecaster = SharedVelocityTrajectoryForecaster(
        world, stream, expected_model_hash=compiled_model_hash(world.model), native_step_budget=320
    )
    prediction = forecaster.forecast(
        controls=32,
        previous_command=np.zeros(3, np.float32),
        parameters=dict(speed_mps=0.8, entry_gap_m=0.9, root_lateral_offset_m=0.0),
    )
    assert prediction["body_failed"]
    assert 1 <= prediction["controls"] < 32
    assert prediction["qpos"].shape[0] == prediction["native_steps"] == prediction["controls"] * 10
    assert forecaster.counters()["attempted_native_steps"] == 320
    assert forecaster.counters()["integrated_native_steps"] == prediction["native_steps"]
    assert forecaster.counters()["future_velocity_nn_calls"] == prediction["controls"]
    assert world.native_steps == stream.neural_calls == 0
