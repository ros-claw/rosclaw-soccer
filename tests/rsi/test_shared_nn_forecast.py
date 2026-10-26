from collections import deque
from types import SimpleNamespace

import mujoco
import numpy as np
import onnxruntime as ort
import pytest
from onnx import TensorProto, helper, numpy_helper
from test_shared_native_motor_world import source_xml

from rosclaw_soccer.rsi.shared_native_motor_world import NativePDParameters, SharedNativeMotorWorld
from rosclaw_soccer.rsi.shared_native_scene import shared_native_scene
from rosclaw_soccer.rsi.shared_nn_forecast import (
    SharedNNForecaster,
    _fork_pending_player,
    copy_sim_policy,
    policy_state_hash,
)
from rosclaw_soccer.rsi.shared_robonaldo_player import SharedRoboNaldoPlayer
from rosclaw_soccer.rsi.shared_target_forecast import compiled_model_hash

MUJOCO_TO_ISAAC = np.arange(29)


class ForecastPolicy:
    def __init__(self, session):
        self.runtime_mode = "sim"
        self.use_body_frame_ball = False
        self.time_step = 0
        self.WARMUP_STEPS = 0
        self.kps, self.kds = np.full(29, 20.0), np.ones(29)
        self.default_q_mj, self.action_scale_mj = np.zeros(29, np.float32), np.ones(29, np.float32)
        self.action_clip_lo_il, self.action_clip_hi_il = np.full(29, -10.0), np.full(29, 10.0)
        self.last_action_il = np.zeros(29, np.float32)
        self.state_cmd, self.policy_output = SimpleNamespace(), SimpleNamespace()
        self.history = deque([np.zeros(29, np.float32)], maxlen=5)
        self.ort_session = session
        self.enters = 0

    def enter(self):
        self.enters += 1

    def run(self):
        self.history.append(self.last_action_il.copy())
        obs = np.zeros((1, 547), np.float32)
        obs[0, :29] = self.state_cmd.q + 0.2 * self.history[-1]
        if hasattr(self, "motion_joint_pos"):
            obs[0, :29] += self.motion_joint_pos[self.time_step]
        out = self.ort_session.run(["actions"], {"obs": obs})[0][0]
        self.last_action_il = out.copy()
        self.policy_output.actions = out.copy()
        self.policy_output.kps, self.policy_output.kds = self.kps.copy(), self.kds.copy()
        self.time_step += 1


def setup(tmp_path, budget=80, fixture_xml=None):
    matrix = np.zeros((547, 29), np.float32)
    matrix[:29] = np.eye(29, dtype=np.float32)
    graph = helper.make_graph(
        [
            helper.make_node("MatMul", ["obs", "weight"], ["linear"]),
            helper.make_node("Add", ["linear", "bias"], ["actions"]),
        ],
        "finite_fixture",
        [helper.make_tensor_value_info("obs", TensorProto.FLOAT, [1, 547])],
        [helper.make_tensor_value_info("actions", TensorProto.FLOAT, [1, 29])],
        initializer=[
            numpy_helper.from_array(matrix, "weight"),
            numpy_helper.from_array(np.full(29, 0.001, np.float32), "bias"),
        ],
    )
    proto = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)], ir_version=10)
    session = ort.InferenceSession(proto.SerializeToString(), providers=["CPUExecutionProvider"])
    xml = shared_native_scene(
        (source_xml() if fixture_xml is None else fixture_xml)
        .replace('name="link0"', 'name="torso_link"')
        .replace("<joint name=", '<joint limited="true" range="-2 2" name='),
        prefixes=("a/",),
    )
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    world = SharedNativeMotorWorld(
        model,
        data,
        players={"a/": NativePDParameters(np.full(29, 20.0), np.ones(29), np.full(29, 25.0))},
        ball_body_name="ball",
    )
    player = SharedRoboNaldoPlayer(
        world, "a/", ForecastPolicy(session), goal=np.array([5.0, 0.0, 0.11])
    )
    first, _ = player.propose()
    world.advance({"a/": first})
    player.commit()
    player.propose()  # Nonzero actual time and nonempty measured software history.
    forecaster = SharedNNForecaster(
        player,
        expected_policy_type=ForecastPolicy,
        expected_model_hash=compiled_model_hash(model),
        native_step_budget=budget,
    )
    return player, forecaster


def test_owned_nn_history_matches_actual_subsequent_multicontrol_and_donor_is_unchanged(tmp_path):
    p, f = setup(tmp_path)
    before = (
        policy_state_hash(p.policy),
        p.world.data.qpos.copy(),
        p.world.data.ctrl.copy(),
        p.world.data.time,
    )
    predicted = f.forecast(controls=4)
    assert before[0] == policy_state_hash(p.policy)
    np.testing.assert_array_equal(before[1], p.world.data.qpos)
    np.testing.assert_array_equal(before[2], p.world.data.ctrl)
    assert p.world.data.time == before[3]
    actual, targets = [], []
    for i in range(4):
        target = p.pending.copy() if i == 0 else p.propose()[0]
        targets.append(target.copy())
        actual.append(p.world.advance({p.prefix: target}))
        p.commit()
    for k in ("qpos", "qvel", "torque", "ball_forces"):
        np.testing.assert_array_equal(predicted[k], np.concatenate([r[k] for r in actual]))
    np.testing.assert_array_equal(predicted["joint_targets"], np.stack(targets))
    assert p.policy.enters == 1
    assert f.counters()["future_nn_calls"] == 3


def test_independent_forecast_copies_and_budget_enforcement(tmp_path):
    p, f = setup(tmp_path, budget=80)
    first, second = f.forecast(controls=4), f.forecast(controls=4)
    np.testing.assert_array_equal(first["qpos"], second["qpos"])
    assert p.world.control_steps == 1
    with pytest.raises(RuntimeError, match="exhausted"):
        f.forecast(controls=1)
    assert f.counters()["integrated_native_steps"] == 80


def test_nested_owned_future_keeps_absolute_clock_without_faking_consumed_steps(tmp_path):
    player, forecaster = setup(tmp_path)
    donor = player.world
    data = mujoco.MjData(donor.model)
    mujoco.mj_copyData(data, donor.model, donor.data)
    owned = SharedNativeMotorWorld(
        donor.model, data, players=donor.parameters, ball_body_name="ball"
    )
    ghost = _fork_pending_player(player, owned, copy_sim_policy(player.policy))
    nested = SharedNNForecaster(
        ghost,
        expected_policy_type=ForecastPolicy,
        expected_model_hash=compiled_model_hash(donor.model),
        native_step_budget=80,
    )
    expected, actual = forecaster.forecast(controls=4), nested.forecast(controls=4)
    assert actual["donor_control_frame"] == expected["donor_control_frame"] == 1
    assert donor.control_steps == 1 and owned.control_steps == 0
    assert owned.data.time == donor.data.time
    for key in ("qpos", "qvel", "torque", "ball_forces", "joint_targets", "policy_ticks"):
        np.testing.assert_array_equal(actual[key], expected[key])
    ghost._forecast_control_origin = 2
    with pytest.raises(ValueError, match="uninterrupted native episode time"):
        nested.forecast(controls=1)
    assert nested.counters()["attempted_native_steps"] == 40


def test_retired_player_cannot_make_new_or_existing_forecasts(tmp_path):
    player, forecaster = setup(tmp_path)
    player.retired = True
    with pytest.raises(ValueError, match="unchanged actual boundary"):
        forecaster.forecast(controls=1)
    assert forecaster.counters()["attempted_native_steps"] == 0
    with pytest.raises(ValueError, match="qualified single-player"):
        SharedNNForecaster(
            player,
            expected_policy_type=ForecastPolicy,
            expected_model_hash=compiled_model_hash(player.world.model),
            native_step_budget=10,
        )


def test_modified_first_action_requires_actual_encoding_and_copied_history_only(tmp_path):
    p, f = setup(tmp_path)
    history = p.policy.last_action_il.copy()
    history[0] += np.float32(0.02)
    target = history.astype(np.float64)
    before = policy_state_hash(p.policy)
    altered = f.forecast(controls=4, first_target=target, first_action_history=history)
    original = f.forecast(controls=4)
    assert before == policy_state_hash(p.policy)
    assert not np.array_equal(altered["joint_targets"][1], original["joint_targets"][1])
    assert not np.array_equal(altered["qpos"], original["qpos"])


def test_invalid_target_is_charged_without_hidden_integration(tmp_path):
    p, f = setup(tmp_path)
    with pytest.raises(ValueError, match="encoding"):
        f.forecast(controls=4, first_target=np.full(29, np.nan))
    assert f.counters()["attempted_native_steps"] == 40
    assert f.counters()["integrated_native_steps"] == 0
    assert p.world.control_steps == 1


def test_unsupported_resource_is_not_deepcopied_or_executed(tmp_path):
    p, f = setup(tmp_path)

    class Resource:
        def __deepcopy__(self, memo):
            raise AssertionError("unknown copy hook must never run")

    p.policy.unqualified_resource = Resource()
    with pytest.raises(ValueError, match="unsupported resource"):
        f.forecast(controls=4)
    assert f.counters()["integrated_native_steps"] == 0


def test_no_pending_boundary_rejected_without_allocating_a_forecast(tmp_path):
    p, f = setup(tmp_path)
    p.world.advance({p.prefix: p.pending})
    p.commit()
    with pytest.raises(ValueError, match="pending"):
        f.forecast(controls=1)
    assert f.counters()["attempted_native_steps"] == 0


def test_preproposal_fork_calls_first_nn_without_reset_or_donor_mutation(tmp_path):
    p, f = setup(tmp_path)
    p.world.advance({p.prefix: p.pending})
    p.commit()
    before = policy_state_hash(p.policy)
    predicted = f.forecast(controls=4, before_proposal=True)
    assert before == policy_state_hash(p.policy)
    assert p.pending is None
    records, targets = [], []
    for _ in range(4):
        target, _ = p.propose()
        targets.append(target)
        records.append(p.world.advance({p.prefix: target}))
        p.commit()
    for k in ("qpos", "qvel", "torque", "ball_forces"):
        np.testing.assert_array_equal(predicted[k], np.concatenate([r[k] for r in records]))
    np.testing.assert_array_equal(predicted["joint_targets"], np.stack(targets))
    assert f.counters()["future_nn_calls"] == 4
    assert p.policy.enters == 1


def test_preproposal_cannot_overwrite_first_action_or_use_pending_history(tmp_path):
    p, f = setup(tmp_path)
    with pytest.raises(ValueError, match="pending"):
        f.forecast(controls=4, before_proposal=True)
    p.world.advance({p.prefix: p.pending})
    p.commit()
    with pytest.raises(ValueError, match="pre-proposal only"):
        f.forecast(controls=4, before_proposal=True, first_target=np.zeros(29))
    assert f.counters()["attempted_native_steps"] == 0


@pytest.mark.parametrize("kind", ["timing", "synergy"])
def test_conditioned_reference_forecast_equals_actual_selected_plan(tmp_path, kind):
    from rosclaw_soccer.rsi.motion_reference_timing import (
        MotionTimingPlan,
        condition_motion_reference,
    )
    from rosclaw_soccer.rsi.motion_synergy_reference import (
        MotionSynergyPlan,
        condition_synergy_reference,
    )

    p, f = setup(tmp_path)
    p.world.advance({p.prefix: p.pending})
    p.commit()
    policy = p.policy
    policy.control_dt, policy.motion_total_steps = 0.02, 100
    t = np.arange(100)[:, None] * 0.02
    policy.motion_joint_pos = np.repeat(t**2 * 0.001, 29, axis=1)
    policy.motion_joint_vel = np.repeat(2 * t * 0.001, 29, axis=1)
    policy.motion_body_pos = np.zeros((100, 2, 3))
    policy.motion_body_quat = np.tile([1.0, 0.0, 0.0, 0.0], (100, 2, 1))
    plan = (
        MotionTimingPlan(2, 24, 3.0)
        if kind == "timing"
        else MotionSynergyPlan(2, ((0.01,) * 29,), ((0.01,), (-0.01,)))
    )
    before = policy_state_hash(policy)
    predicted = f.forecast(controls=4, before_proposal=True, reference_plan=plan)
    assert before == policy_state_hash(policy)
    if kind == "timing":
        condition_motion_reference(policy, plan)
    else:
        condition_synergy_reference(policy, plan)
    records = []
    for _ in range(4):
        target, _ = p.propose()
        records.append(p.world.advance({p.prefix: target}))
        p.commit()
    for k in ("qpos", "qvel", "torque", "ball_forces"):
        np.testing.assert_array_equal(predicted[k], np.concatenate([r[k] for r in records]))
