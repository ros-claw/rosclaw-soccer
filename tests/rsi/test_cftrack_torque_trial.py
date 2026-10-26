from types import SimpleNamespace

import numpy as np
import pytest

from rosclaw_soccer.rsi.cftrack_torque_trial import TorqueTrialCapture, complete_torque_trial


def test_incomplete_or_failed_feedback_never_counts_as_complete():
    result = dict(
        controls=160,
        native_steps=1600,
        neural_calls=160,
        layer_commands=1600,
        layer_feedback=1600,
        incomplete_control_native_steps=0,
        runtime_failure=None,
        body_failed=False,
    )
    assert complete_torque_trial(result)
    for key in ("controls", "native_steps", "neural_calls", "layer_commands", "layer_feedback"):
        assert not complete_torque_trial({**result, key: result[key] - 1})
    assert not complete_torque_trial({**result, "runtime_failure": {"type": "fault"}})
    assert not complete_torque_trial({**result, "body_failed": True})
    assert not complete_torque_trial({**result, "incomplete_control_native_steps": 1})


def test_capture_preserves_original_outputs_and_does_not_alias_live_arrays():
    out = np.zeros((1, 29), np.float32)
    session = SimpleNamespace(run=lambda outputs, inputs: [out])
    capture = TorqueTrialCapture(session)
    inputs = {"obs": np.ones((1, 1244), np.float32), "time_step": np.zeros((1, 1), np.float32)}
    assert capture.run(None, inputs)[0] is out
    out[:] = 3
    inputs["obs"][:] = 4
    assert not capture.actions[0].any()
    assert (capture.observations[0] == 1).all()


def test_capture_does_not_call_nn_beyond_budget_or_with_invalid_input():
    def forbidden(*args):
        raise AssertionError("must reject before NN")

    capture = TorqueTrialCapture(SimpleNamespace(run=forbidden))
    inputs = {"obs": np.zeros((1, 1244), np.float32), "time_step": np.ones((1, 1), np.float32)}
    with pytest.raises(ValueError):
        capture.run(None, inputs)
    inputs["time_step"][:] = 0
    inputs["obs"][0, 0] = float("nan")
    with pytest.raises(ValueError):
        capture.run(None, inputs)
    inputs["obs"][:] = 0
    capture.observations = [np.zeros((1, 1244))] * 160
    with pytest.raises(ValueError):
        capture.run(None, inputs)


def test_cold_trial_records_real_native_feedback_with_fresh_fixture(tmp_path):
    import mujoco
    from test_shared_cftrack_forecast import Policy, setup
    from test_shared_native_motor_world import source_xml

    from rosclaw_soccer.rsi.cftrack_torque_intent import CFtrackTorqueIntentPlan
    from rosclaw_soccer.rsi.cftrack_torque_replay import replay_torque_trial
    from rosclaw_soccer.rsi.cftrack_torque_trial import run_cold_torque_trial
    from rosclaw_soccer.rsi.omnicontact_policy_copy import cftrack_state_hash
    from rosclaw_soccer.rsi.shared_native_motor_world import (
        NativePDParameters,
        SharedNativeMotorWorld,
    )
    from rosclaw_soccer.rsi.shared_native_scene import shared_native_scene
    from rosclaw_soccer.rsi.shared_omnicontact_player import SharedOmniContactPlayer
    from rosclaw_soccer.sim.contracts import hash_json

    old, _ = setup(tmp_path, cold=True)
    xml = (
        source_xml()
        .replace('name="link0"', 'name="torso_link"')
        .replace('name="link1"', 'name="left_ankle_roll_link"')
        .replace('name="link2"', 'name="right_ankle_roll_link"')
    )
    model = mujoco.MjModel.from_xml_string(shared_native_scene(xml, prefixes=("a/",)))
    model.opt.gravity[:] = 0.0
    data = mujoco.MjData(model)
    data.qpos[model.joint("a/root").qposadr[0] + 2] = 0.77
    mujoco.mj_forward(model, data)
    params = NativePDParameters(np.full(29, 20.0), np.ones(29), np.full(29, 25.0))
    world = SharedNativeMotorWorld(model, data, players={"a/": params}, ball_body_name="ball")
    policy = Policy(old.policy.ort_session)
    player = SharedOmniContactPlayer(
        world, "a/", policy, expected_policy_type=Policy, goal=np.array([5.0, 0.0, 0.11])
    )
    fractions = [0.0] * 29
    fractions[7] = 0.01
    plan = CFtrackTorqueIntentPlan(
        0, 24, tuple(fractions), hash_json(player.layout.profile()), cftrack_state_hash(policy)
    )
    original_session = policy.ort_session
    raw, report = run_cold_torque_trial(player, plan)
    assert complete_torque_trial(report), report
    assert report["actual_changed_torque_entries"] > 0
    assert raw["qpos"].shape == (1600, model.nq)
    assert raw["neural_obs"].shape == (160, 1, 1244)
    assert policy.ort_session is original_session
    replay_data = mujoco.MjData(model)
    replay_data.qpos[:] = raw["initial_qpos"]
    replay_data.qvel[:] = raw["initial_qvel"]
    mujoco.mj_forward(model, replay_data)
    verified = replay_torque_trial(
        model,
        replay_data,
        player.layout,
        raw,
        plan,
        kp=world.parameters["a/"].stiffness,
        kd=world.parameters["a/"].damping,
        effort=world.parameters["a/"].effort,
    )
    assert verified["values_exact"] and verified["native_steps"] == 1600
    broken = dict(raw, parent_torque=raw["parent_torque"].copy())
    broken["parent_torque"][0, 0] += 1
    replay_data = mujoco.MjData(model)
    replay_data.qpos[:] = raw["initial_qpos"]
    replay_data.qvel[:] = raw["initial_qvel"]
    mujoco.mj_forward(model, replay_data)
    with pytest.raises(ValueError, match="parent PD mismatch"):
        replay_torque_trial(
            model,
            replay_data,
            player.layout,
            broken,
            plan,
            kp=world.parameters["a/"].stiffness,
            kd=world.parameters["a/"].damping,
            effort=world.parameters["a/"].effort,
        )
