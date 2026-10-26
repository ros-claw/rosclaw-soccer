import copy
import threading

import mujoco
import numpy as np
import onnxruntime as ort
import pytest
from onnx import TensorProto, helper, numpy_helper
from test_omnicontact_policy_copy import Kinematics, Record
from test_shared_nn_forecast import setup as original_setup
from test_shared_omnicontact_player import Policy as BasePolicy

from rosclaw_soccer.rsi.cftrack_motor_residual_plan import CFtrackMotorResidualPlan
from rosclaw_soccer.rsi.omnicontact_policy_copy import cftrack_state_hash
from rosclaw_soccer.rsi.shared_cftrack_forecast import SharedCFtrackForecaster
from rosclaw_soccer.rsi.shared_native_motor_world import SharedNativeMotorWorld
from rosclaw_soccer.rsi.shared_omnicontact_player import SharedOmniContactPlayer
from rosclaw_soccer.rsi.shared_target_forecast import compiled_model_hash


class Policy(BasePolicy):
    def __init__(self, session):
        super().__init__()
        self.ort_session = session
        self.state_cmd, self.policy_output = Record(), Record()
        self.async_stage_pending = self.async_stage_error = None
        self.async_stage_lock = threading.Lock()
        self.obs_history_buffer = np.zeros((5, 141), np.float32)
        self.action = np.zeros(29, np.float32)
        self.enters = 0
        joints = '<body><joint/><geom type="sphere" size=".01" mass=".01"/>' * 29 + "</body>" * 29
        model = mujoco.MjModel.from_xml_string(
            '<mujoco><worldbody><body><freejoint/><geom type="sphere" size=".1" mass="1"/>'
            + joints
            + "</body></worldbody></mujoco>"
        )
        self.kinematics = Kinematics()
        self.kinematics.model, self.kinematics.data = model, mujoco.MjData(model)

    def enter(self):
        super().enter()
        self.enters += 1

    def run(self):
        self.obs_history_buffer = np.roll(self.obs_history_buffer, -1, axis=0)
        self.obs_history_buffer[-1, :29] = self.action
        obs = np.zeros((1, 1244), np.float32)
        obs[0, :29] = self.state_cmd.q + 0.2 * self.action
        if hasattr(self, "ref_right_ankle_future_pos"):
            obs[0, :3] += 0.01 * self.ref_right_ankle_future_pos[self.counter_step]
        self.action = self.ort_session.run(
            None, {"obs": obs, "time_step": np.zeros((1, 1), np.float32)}
        )[0][0]
        self.policy_output.actions = self.action.copy()
        self.policy_output.kps, self.policy_output.kds = self.kps_lab, self.kds_lab
        self.counter_step += 1
        self.kinematics.data.qpos[:7] = np.r_[self.state_cmd.base_pos, self.state_cmd.base_quat]
        self.kinematics.data.qpos[7:] = self.state_cmd.q
        mujoco.mj_forward(self.kinematics.model, self.kinematics.data)


def setup(tmp_path, *, physical_range=None):
    old, _ = original_setup(tmp_path)
    model = old.world.model
    if physical_range is not None:
        model = copy.copy(model)
        model.jnt_limited[old.layout.joints] = True
        model.jnt_range[old.layout.joints] = [-physical_range, physical_range]
    data = mujoco.MjData(model)
    mujoco.mj_copyData(data, model, old.world.data)
    world = SharedNativeMotorWorld(model, data, players=old.world.parameters, ball_body_name="ball")
    weight = np.zeros((1244, 29), np.float32)
    weight[:29] = np.eye(29, dtype=np.float32)
    graph = helper.make_graph(
        [
            helper.make_node("MatMul", ["obs", "weight"], ["linear"]),
            helper.make_node("Add", ["linear", "bias"], ["actions"]),
        ],
        "CFtrack_fixture",
        [
            helper.make_tensor_value_info("obs", TensorProto.FLOAT, [1, 1244]),
            helper.make_tensor_value_info("time_step", TensorProto.FLOAT, [1, 1]),
        ],
        [helper.make_tensor_value_info("actions", TensorProto.FLOAT, [1, 29])],
        initializer=[
            numpy_helper.from_array(weight, "weight"),
            numpy_helper.from_array(np.full(29, 0.001, np.float32), "bias"),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)], ir_version=10)
    session = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    player = SharedOmniContactPlayer(
        world, "a/", Policy(session), expected_policy_type=Policy, goal=np.array([5.0, 0.0, 0.11])
    )
    forecast = SharedCFtrackForecaster(
        player,
        expected_policy_type=Policy,
        expected_model_hash=compiled_model_hash(world.model),
        native_step_budget=80,
    )
    return player, forecast


def test_private_cftrack_nn_and_fk_prediction_exact_actual_without_enter_reset(tmp_path):
    p, f = setup(tmp_path)
    before = cftrack_state_hash(p.policy)
    initial = p.world.data.qpos.copy()
    predicted = f.forecast(controls=4)
    assert predicted["runtime_failure"] is None and predicted["controls"] == 4
    assert cftrack_state_hash(p.policy) == before and p.policy.enters == 1
    np.testing.assert_array_equal(p.world.data.qpos, initial)
    assert p.world.control_steps == 0
    rows = []
    for _ in range(4):
        target, _ = p.propose()
        rows.append(p.world.advance({p.prefix: target}))
        p.commit()
    for key in ("qpos", "qvel", "torque", "ball_forces"):
        np.testing.assert_array_equal(predicted[key], np.concatenate([r[key] for r in rows]))
    assert f.counters()["future_nn_calls"] == 4 and p.policy.enters == 1
    # Fork a nonempty software history and nonzero actual native clock too.
    before = cftrack_state_hash(p.policy)
    predicted = f.forecast(controls=4)
    assert predicted["runtime_failure"] is None and predicted["controls"] == 4
    assert cftrack_state_hash(p.policy) == before
    for i in range(4):
        target, _ = p.propose()
        np.testing.assert_array_equal(target, predicted["joint_targets"][i])
        row = p.world.advance({p.prefix: target})
        p.commit()
        np.testing.assert_array_equal(row["qpos"], predicted["qpos"][i * 10 : (i + 1) * 10])
    assert f.counters()["integrated_native_steps"] == 80
    with pytest.raises(ValueError, match="budget"):
        f.forecast(controls=1)
    assert f.counters()["attempted_native_steps"] == 80


def test_pending_or_failed_player_is_not_forked(tmp_path):
    p, f = setup(tmp_path)
    p.propose()
    with pytest.raises(ValueError, match="pre-proposal"):
        f.forecast(controls=1)
    assert f.counters()["forecast_calls"] == 0


def test_private_failure_preserves_partial_raw_and_full_reservation(tmp_path, monkeypatch):
    p, f = setup(tmp_path)
    before = cftrack_state_hash(p.policy)
    original = Policy.run

    def fail_second(self):
        if self.counter_step == 1:
            raise ValueError("injected private prediction failure")
        original(self)

    monkeypatch.setattr(Policy, "run", fail_second)
    predicted = f.forecast(controls=4)
    assert predicted["controls"] == 1 and predicted["qpos"].shape == (10, 43)
    assert predicted["runtime_failure"]["message"] == "injected private prediction failure"
    assert cftrack_state_hash(p.policy) == before and p.world.native_steps == 0
    assert f.counters()["attempted_native_steps"] == 40
    assert f.counters()["integrated_native_steps"] == 10
    with pytest.raises(ValueError, match="budget"):
        f.forecast(controls=5)


def motor_plan(p, *, amount=0.01, peak=2):
    delta = [0.0] * 29
    delta[8] = float(amount)
    return CFtrackMotorResidualPlan(
        p.policy.counter_step,
        p.policy.counter_step,
        peak,
        tuple(delta),
        cftrack_state_hash(p.policy),
    )


@pytest.mark.parametrize("amount", [0.0, 0.01])
def test_motor_future_matches_actual_with_original_nn_and_inherited_nonzero_clock(tmp_path, amount):
    p, f = setup(tmp_path, physical_range=1.0)
    plan = motor_plan(p, amount=amount)
    before = cftrack_state_hash(p.policy)
    future = f.forecast(controls=4, motor_plan=plan)
    assert future["runtime_failure"] is None and future["controls"] == 4
    assert cftrack_state_hash(p.policy) == before and p.motor_plan is None
    if amount:
        assert np.any(future["joint_targets"] != future["nn_decoded_targets"])
    else:
        np.testing.assert_array_equal(future["joint_targets"], future["nn_decoded_targets"])
    p.install_motor_plan(plan)
    for tick in range(4):
        target, observed = p.propose()
        np.testing.assert_array_equal(target, future["joint_targets"][tick])
        np.testing.assert_array_equal(
            observed["nn_decoded_target"], future["nn_decoded_targets"][tick]
        )
        row = p.world.advance({p.prefix: target})
        p.commit()
        np.testing.assert_array_equal(row["qpos"], future["qpos"][tick * 10 : (tick + 1) * 10])
    second = f.forecast(controls=4)
    assert second["runtime_failure"] is None and second["controls"] == 4
    for tick in range(4):
        target, _ = p.propose()
        np.testing.assert_array_equal(target, second["joint_targets"][tick])
        row = p.world.advance({p.prefix: target})
        p.commit()
        np.testing.assert_array_equal(row["qvel"], second["qvel"][tick * 10 : (tick + 1) * 10])


def test_rejected_post_nn_motor_target_counts_nn_but_never_integrates_that_target(tmp_path):
    p, f = setup(tmp_path, physical_range=0.01)
    before = cftrack_state_hash(p.policy)
    future = f.forecast(controls=4, motor_plan=motor_plan(p, amount=0.15, peak=4))
    assert future["controls"] == 1 and future["native_steps"] == 10
    assert "no clipping" in future["runtime_failure"]["message"]
    assert f.counters()["future_nn_calls"] == 2 and f.counters()["attempted_native_steps"] == 40
    assert p.world.native_steps == 0 and cftrack_state_hash(p.policy) == before


def test_donor_mismatch_and_motor_stacking_reject_before_physical_allocation(tmp_path):
    from dataclasses import replace

    p, f = setup(tmp_path, physical_range=1.0)
    plan = motor_plan(p)
    with pytest.raises(ValueError, match="donor"):
        f.forecast(controls=4, motor_plan=replace(plan, donor_policy_hash="sha256:" + "0" * 64))
    assert f.counters()["attempted_native_steps"] == 0
    p.install_motor_plan(plan)
    with pytest.raises(ValueError):
        p.install_motor_plan(plan)
    with pytest.raises(ValueError):
        f.forecast(controls=4, motor_plan=plan)
    assert f.counters()["attempted_native_steps"] == 0
