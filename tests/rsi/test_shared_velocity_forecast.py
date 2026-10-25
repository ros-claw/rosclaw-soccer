"""Real fixture MuJoCo + synthetic CPU NN; not trained G1 performance evidence."""

import mujoco
import numpy as np
import pytest
from test_shared_native_motor_world import source_xml
from test_velocity_stream_branch import synthetic_stream

from rosclaw_soccer.rsi.shared_native_motor_world import NativePDParameters, SharedNativeMotorWorld
from rosclaw_soccer.rsi.shared_native_scene import shared_native_scene
from rosclaw_soccer.rsi.shared_target_forecast import compiled_model_hash
from rosclaw_soccer.rsi.shared_velocity_forecast import SharedVelocityForecaster


def setup():
    stream = synthetic_stream()
    model = mujoco.MjModel.from_xml_string(shared_native_scene(source_xml(), prefixes=("a/",)))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    world = SharedNativeMotorWorld(
        model,
        data,
        players={
            "a/": NativePDParameters(
                np.array(stream.params.stiffness),
                np.array(stream.params.damping),
                np.full(29, 25.0),
            )
        },
        ball_body_name="ball",
    )
    predictor = SharedVelocityForecaster(
        world, stream, expected_model_hash=compiled_model_hash(model), native_step_budget=30
    )
    return world, stream, predictor


def test_copied_nn_history_and_native_one_control_are_bitwise_actual():
    world, stream, predictor = setup()
    command = np.array([0.4, 0.1, -0.1], np.float32)
    lo = world.layouts["a/"]
    for frame in range(3):
        q, v = lo.canonical(world.data)
        before = world.data.qpos.copy(), world.data.qvel.copy(), float(world.data.time)
        predicted = predictor.forecast(command)
        np.testing.assert_array_equal(world.data.qpos, before[0])
        np.testing.assert_array_equal(world.data.qvel, before[1])
        assert float(world.data.time) == before[2]
        assert stream.next_frame == frame and stream.neural_calls == frame
        actual_target = stream.propose(
            q, v, lo.angular_velocity_body(world.data), command, frame=frame
        )
        np.testing.assert_array_equal(actual_target, predicted["joint_target"])
        np.testing.assert_array_equal(stream.last_observation, predicted["observation"])
        np.testing.assert_array_equal(stream.last_action, predicted["action"])
        assert float(stream.phase) == predicted["phase_after"]
        actual = world.advance({"a/": actual_target})
        for key in ("qpos", "qvel", "torque", "ball_forces"):
            np.testing.assert_array_equal(actual[key], predicted[key])
    assert predictor.counters()["future_velocity_nn_calls"] == 3
    assert predictor.counters()["integrated_native_steps"] == 30
    with pytest.raises(RuntimeError, match="exhausted"):
        predictor.forecast(command)
    assert predictor.counters()["future_velocity_nn_calls"] == 3


def test_switched_pd_is_not_a_velocity_future():
    world, stream, predictor = setup()
    world.switch_pd_profile(
        "a/", NativePDParameters(np.full(29, 20.0), np.ones(29), np.full(29, 25.0))
    )
    with pytest.raises(ValueError, match="switched"):
        predictor.forecast(np.zeros(3, np.float32))
    assert stream.neural_calls == world.native_steps == 0


def test_pending_actual_nn_proposal_is_not_a_completed_world_boundary():
    world, stream, predictor = setup()
    lo = world.layouts["a/"]
    q, v = lo.canonical(world.data)
    stream.propose(q, v, lo.angular_velocity_body(world.data), np.zeros(3, np.float32), frame=0)
    with pytest.raises(ValueError, match="completed"):
        predictor.forecast(np.zeros(3, np.float32))
    assert stream.neural_calls == 1 and world.native_steps == 0
