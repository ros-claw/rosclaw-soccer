"""Toy dynamics/control clocks only; no actual G1 skill qualification."""

import mujoco
import numpy as np
import pytest

from rosclaw_soccer.physics.g1_early_feedback_forecast import G1EarlyFeedbackForecaster
from rosclaw_soccer.rsi.feedback_branch_control import G1FeedbackBranchControl
from rosclaw_soccer.sim.contracts import G1_HARD_TORQUE_LIMITS, hash_bytes
from tests.physics.test_g1_feedback_forecast import control, world


class FakeTracker(int):
    def neural_observation(self):
        return {}


def fixture(budget=80, *, reuse=False):
    model, data, _ = world()
    binary = np.empty(mujoco.mj_sizeModel(model), dtype=np.uint8)
    mujoco.mj_saveModel(model, None, binary)
    return (
        model,
        data,
        G1EarlyFeedbackForecaster(
            model,
            world_hash=hash_bytes(binary.tobytes()),
            kp=np.full(29, 10.0),
            kd=np.full(29, 0.2),
            max_forecast_steps=budget,
            reuse_owned_updates=reuse,
        ),
    )


def fake_controller(monkeypatch, model, data, *, failure=None):
    calls, forks = [], []

    def init(self, model, **kw):
        forks.append(kw)
        assert kw["tracker"] == kw["frame"]
        assert kw["decoder"] == kw["frame"]
        assert kw["ik"] == kw["frame"]
        assert kw["body_prefix"]["frame"] == kw["frame"]
        np.testing.assert_array_equal(kw["previous_motor_delta_rad"], kw["frame"] * 0.001)
        self.model, self.frame, self.consumed = model, kw["frame"], False

    def next_target(self, d, forces):
        assert d is not data and d.model is model
        assert d.time == pytest.approx((self.frame + 1) * 0.02)
        assert not self.consumed
        assert forces.shape == (6,) and np.all(forces >= 0)
        self.consumed = True
        calls.append(self.frame)
        if self.frame == failure:
            raise ValueError("synthetic next-frame failure")
        self.tracker = FakeTracker(self.frame + 1)
        self.decoder = self.ik = self.frame + 1
        self.policy = {"frozen": True}
        self.body = {"frame": self.frame + 1}
        target = np.full(29, 0.1 * (self.frame - 29))
        return target, dict(
            joint_target_rad=target.copy(), motor_delta_rad=np.full(12, (self.frame + 1) * 0.001)
        )

    monkeypatch.setattr(G1FeedbackBranchControl, "__init__", init)
    monkeypatch.setattr(G1FeedbackBranchControl, "next_target", next_target)
    return calls, forks


@pytest.mark.parametrize("offset", [0.0, 1.0, -1.0])
def test_full80ms_recomputes_three_targets_and_removes_offset_for_last20ms(monkeypatch, offset):
    model, donor, f = fixture()
    calls, forks = fake_controller(monkeypatch, model, donor)
    spec = mujoco.mjtState.mjSTATE_INTEGRATION
    before = np.empty(mujoco.mj_stateSize(model, spec))
    mujoco.mj_getState(model, donor, before, spec)
    prediction = f.forecast(
        donor, np.zeros(29), control=control(model), early_offset_nm=np.full(29, offset)
    )
    assert calls == [30, 31, 32]
    assert [kw["frame"] for kw in forks] == [31, 32]
    assert len(prediction.next_target_records) == 3
    assert prediction.arrays["completed_contact_peak_force_n"].shape == (40, 6)
    np.testing.assert_array_equal(prediction.arrays["requested_offset_nm"][:30], offset)
    np.testing.assert_array_equal(prediction.arrays["requested_offset_nm"][30:], 0)
    expected = mujoco.MjData(model)
    mujoco.mj_copyData(expected, model, donor)
    hard = np.asarray(G1_HARD_TORQUE_LIMITS)
    for step in range(40):
        target = np.full(29, 0.1 * (step // 10))
        raw = (target - expected.qpos[7:36]) * 10.0 - expected.qvel[6:35] * 0.2
        requested = raw + offset if step < 30 and offset else raw
        expected.ctrl[:] = np.clip(requested, -hard, hard)
        mujoco.mj_step(model, expected)
        np.testing.assert_array_equal(
            prediction.arrays["canonical_qpos_after_substep"][step], expected.qpos
        )
        np.testing.assert_array_equal(
            prediction.arrays["canonical_qvel_after_substep"][step], expected.qvel
        )
        np.testing.assert_array_equal(prediction.arrays["applied_control_nm"][step], expected.ctrl)
    after = np.empty_like(before)
    mujoco.mj_getState(model, donor, after, spec)
    assert before.tobytes() == after.tobytes()
    again = f.forecast(
        donor, np.zeros(29), control=control(model), early_offset_nm=np.full(29, offset)
    )
    assert again.end_integration_state.tobytes() == prediction.end_integration_state.tobytes()
    assert f.attempted_steps == f.integrated_steps == 80
    assert f.next_target_attempts == f.next_target_completed == 6
    with pytest.raises(RuntimeError, match="budget"):
        f.forecast(donor, np.zeros(29), control=control(model), early_offset_nm=np.zeros(29))


@pytest.mark.parametrize("failure,steps,completed", [(30, 10, 0), (31, 20, 1), (32, 30, 2)])
def test_failed_controller_is_terminal_and_preserves_actual_charges(
    monkeypatch, failure, steps, completed
):
    model, donor, f = fixture()
    fake_controller(monkeypatch, model, donor, failure=failure)
    before = donor.qpos.copy()
    with pytest.raises(ValueError, match="synthetic"):
        f.forecast(donor, np.zeros(29), control=control(model), early_offset_nm=np.zeros(29))
    assert f.attempted_steps == f.integrated_steps == steps
    assert f.next_target_completed == completed
    assert f.next_target_attempts == completed + 1
    np.testing.assert_array_equal(donor.qpos, before)
    with pytest.raises(ValueError, match="unused causal"):
        f.forecast(donor, np.zeros(29), control=control(model), early_offset_nm=np.zeros(29))
    assert f.attempted_steps == steps


@pytest.mark.parametrize("frame", [29, 37, 38, 39, 57, 58, 59, 295, 298, True])
def test_rejects_refresh_crossing_or_invalid_frame_before_any_physics(frame):
    model, donor, f = fixture()
    c = control(model)
    c.frame, donor.time = frame, frame * 0.02
    with pytest.raises(ValueError, match="unused causal"):
        f.forecast(donor, np.zeros(29), control=c, early_offset_nm=np.zeros(29))
    assert f.attempted_steps == f.forecast_attempts == 0


@pytest.mark.parametrize("budget", [0, 20, 41, 600040, True, 40.0])
def test_rejects_invalid_budget(budget):
    with pytest.raises(ValueError, match="whole80ms"):
        fixture(budget)


@pytest.mark.parametrize("value", [None, 0, 1, "true"])
def test_owned_update_option_requires_explicit_boolean(value):
    with pytest.raises(ValueError, match="whole80ms"):
        fixture(reuse=value)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), 13.0])
def test_rejects_invalid_action_before_physics(value):
    model, donor, f = fixture()
    with pytest.raises(ValueError, match="finite|bounded"):
        f.forecast(donor, np.zeros(29), control=control(model), early_offset_nm=np.full(29, value))
    assert f.attempted_steps == 0
