"""State-machine tests with toy MuJoCo state and synthetic forecast/Jacobians.

No real G1 forecasting, learned policy, physical success or promotion proof.
"""

from dataclasses import replace
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from rosclaw_soccer.rsi import postcontact_exit_controller as mod
from tests.physics.test_g1_feedback_forecast import world
from tests.rsi.test_early_feedback_impact_score import fixture


def candidate(name, offset=None, vx=0.5, unsafe=False):
    offset = np.zeros(29) if offset is None else offset
    f, _ = fixture(first=None, offset=offset)
    f.arrays["canonical_qvel_after_substep"][:, 35:38] = [vx, 0, 0]
    f.arrays["nonfoot_robot_contact_over1n_after_substep"][:] = unsafe
    targets = tuple(dict(target=np.zeros(29)) for _ in range(3))
    neural = tuple(dict(token=np.zeros(64, dtype=np.float32)) for _ in range(3))
    f = replace(f, next_target_records=targets, next_neural_records=neural)
    s = mod.postcontact_exit_score(
        f, offset, previous_contact_forces=np.array([2.0, 0, 0, 0, 0, 0])
    )
    return mod.PostContactExitCandidate(name, offset, f, s)


def bank(*, unsafe=False):
    offset = np.zeros(29)
    offset[3] = 0.1
    return [
        candidate("baseline", unsafe=unsafe),
        candidate("forward_minus", offset, vx=1.2, unsafe=unsafe),
        candidate("repeat_zero", unsafe=unsafe),
    ]


def test_improving_exit_selected_and_ties_or_all_unsafe_are_zero():
    assert mod.select_exit_candidate(bank()) == 1
    assert mod.select_exit_candidate(bank(unsafe=True)) == 0
    b = bank()
    b[1] = candidate("forward_minus", b[1].offset_nm, vx=0.5)
    assert mod.select_exit_candidate(b) == 0


@pytest.mark.parametrize(
    "bad",
    [
        "donor",
        "repeat_state",
        "repeat_raw",
        "repeat_target",
        "repeat_neural",
        "repeat_score",
        "counter",
        "nan",
    ],
)
def test_shared_donor_and_independent_repeat_checks(bad):
    b = bank()
    if bad == "donor":
        b[1] = replace(
            b[1], forecast=replace(b[1].forecast, donor_integration_state_hash="different")
        )
    elif bad == "repeat_state":
        b[-1].forecast.end_integration_state[0] = 1
    elif bad == "repeat_raw":
        b[-1].forecast.arrays["canonical_qvel_after_substep"][0, 0] = 1
    elif bad == "repeat_target":
        b[-1].forecast.next_target_records[2]["target"][0] = 1
    elif bad == "repeat_neural":
        b[-1].forecast.next_neural_records[1]["token"][0] = 1
    elif bad == "repeat_score":
        b[-1].score["proxy_cost"] += 0.1
    elif bad == "counter":
        b[-1] = replace(b[-1], forecast=replace(b[-1].forecast, next_target_records=()))
    else:
        b[1].score["proxy_cost"] = np.nan
    with pytest.raises(ValueError):
        mod.select_exit_candidate(b)


@pytest.fixture
def synthetic(monkeypatch):
    class Forecast:
        def __init__(self, model, **kw):
            assert kw["max_forecast_steps"] == 560
            self.attempted_steps = self.integrated_steps = 0
            self._scratch = mujoco.MjData(model)

        def forecast(self, data, target, *, control, early_offset_nm):
            assert control["frame"] * 0.02 == data.time
            self.attempted_steps += 40
            self.integrated_steps += 40
            return candidate(
                "synthetic", early_offset_nm, vx=1.2 if np.any(early_offset_nm) else 0.5
            ).forecast

    monkeypatch.setattr(mod, "G1EarlyFeedbackForecaster", Forecast)
    monkeypatch.setattr(
        mod,
        "G1JointTaskKinematicSampler",
        lambda *a, **kw: SimpleNamespace(sample=lambda data: None),
    )
    monkeypatch.setattr(mod, "G1FeedbackBranchControl", lambda *a, **kw: kw)
    offset = np.zeros(29)
    offset[3] = 0.1
    monkeypatch.setattr(
        mod,
        "impact_torque_candidates",
        lambda sample, side: (("baseline", np.zeros(29)), ("forward_minus", offset.copy())),
    )
    model, _, _ = world()
    data = mujoco.MjData(model)
    data.qpos[:7] = [0, 0, 0.793, 1, 0, 0, 0]
    c = mod.PostContactExitController(model, world_hash="synthetic", kp=np.ones(29), kd=np.ones(29))
    return c, data


def call(c, data, frame, forces=None):
    data.time = frame * 0.02
    return c.plan(
        data,
        np.zeros(29),
        frame=frame,
        previous_contact_forces=np.zeros(6) if forces is None else forces,
        tracker=None,
        decoder=None,
        ik=None,
        policy={},
        body_prefix={},
        previous_motor_delta_rad=np.zeros(12),
    )


def test_wait_then_single_bank_three_periods_then_never_repeat(synthetic):
    c, d = synthetic
    for f in range(41):
        assert not np.any(call(c, d, f).offset_nm)
    force = np.array([2.0, 0, 0, 0, 0, 0])
    out = call(c, d, 41, force)
    assert out.report["selected_name"] == "forward_minus" and out.report["forecast_substeps"] == 120
    assert out.report["trigger_frame"] == 41 and len(out.candidates) == 3
    held = out.offset_nm.copy()
    out.offset_nm[:] = 0
    for f in (42, 43):
        np.testing.assert_array_equal(call(c, d, f).offset_nm, held)
    for f in range(44, 300):
        out = call(c, d, f, force)
        assert not np.any(out.offset_nm) and not out.candidates
    assert c.review()["forecast_integrated_steps"] == 120
    assert not c.review()["teacher_qualified"]


@pytest.mark.parametrize("bad", ["height", "nonfoot"])
def test_observed_unsafe_cancels_without_resuming(synthetic, bad):
    c, d = synthetic
    for f in range(41):
        call(c, d, f)
    assert np.any(call(c, d, 41, np.array([2.0, 0, 0, 0, 0, 0])).offset_nm)
    force = np.zeros(6)
    if bad == "height":
        d.qpos[2] = 0.64
    else:
        force[4] = 2
    assert not np.any(call(c, d, 42, force).offset_nm)
    d.qpos[2] = 0.793
    assert not np.any(call(c, d, 43, np.array([2.0, 0, 0, 0, 0, 0])).offset_nm)
    assert c.cancelled


@pytest.mark.parametrize("frame", [197, 198, 199, 295, 296, 299])
def test_no_replan_and_late_contact_window_cannot_expand_budget(synthetic, frame):
    c, d = synthetic
    for f in range(frame):
        call(c, d, f)
    out = call(c, d, frame, np.array([2.0, 0, 0, 0, 0, 0]))
    assert out.report["status"] == "CONTACT_OUTSIDE_NO_REPLAN80MS_WINDOW_ZERO_ONLY"
    assert not np.any(out.offset_nm) and c.forecaster.attempted_steps == 0
    for f in range(frame + 1, 300):
        assert not np.any(call(c, d, f, np.array([2.0, 0, 0, 0, 0, 0])).offset_nm)


@pytest.mark.parametrize("bad", ["clock", "repeat", "nan", "negative", "dtype", "initial", "state"])
def test_invalid_input_is_sticky_terminal(synthetic, bad):
    c, d = synthetic
    force = np.zeros(6)
    frame = 0
    if bad == "clock":
        frame = 1
    elif bad == "repeat":
        call(c, d, 0)
    elif bad == "nan":
        force[0] = np.nan
    elif bad == "negative":
        force[0] = -1
    elif bad == "dtype":
        force = force.astype(np.float32)
    elif bad == "initial":
        force[0] = 2
    else:
        d.qvel[0] = np.nan
    with pytest.raises(ValueError):
        call(c, d, frame, force)
    assert c.failed and not c.forecaster.attempted_steps
    with pytest.raises(ValueError, match="cannot resume"):
        call(c, d, c.next_frame)


def test_real_toy_full560_step_bank_three_feedback_updates_and_unchanged_donor(monkeypatch):
    """Actual toy physics; prior contact and kinematics/NN explicitly synthetic."""
    from rosclaw_soccer.sim.contracts import hash_bytes
    from tests.physics.test_g1_early_feedback_forecast import fake_controller
    from tests.rsi.test_impact_torque_screen import sample

    model, data, _ = world()
    binary = np.zeros(mujoco.mj_sizeModel(model), dtype=np.uint8)
    mujoco.mj_saveModel(model, None, binary)
    monkeypatch.setattr(
        mod,
        "G1JointTaskKinematicSampler",
        lambda *a, **kw: SimpleNamespace(sample=lambda data: sample()),
    )
    calls, _ = fake_controller(monkeypatch, model, data)
    c = mod.PostContactExitController(
        model, world_hash=hash_bytes(binary.tobytes()), kp=np.full(29, 10.0), kd=np.full(29, 0.2)
    )
    for f in range(30):
        call(c, data, f)
    data.time = 0.6
    spec = mujoco.mjtState.mjSTATE_INTEGRATION
    before = np.empty(mujoco.mj_stateSize(model, spec))
    mujoco.mj_getState(model, data, before, spec)
    out = c.plan(
        data,
        np.zeros(29),
        frame=30,
        previous_contact_forces=np.array([2.0, 0, 0, 0, 0, 0]),
        tracker=30,
        decoder=30,
        ik=30,
        policy={"frozen": True},
        body_prefix={"frame": 30},
        previous_motor_delta_rad=np.full(12, 0.03),
    )
    after = np.empty_like(before)
    mujoco.mj_getState(model, data, after, spec)
    np.testing.assert_array_equal(before, after)
    assert len(out.candidates) == 14 and len(calls) == 42
    assert calls == [30, 31, 32] * 14
    assert c.forecaster.attempted_steps == c.forecaster.integrated_steps == 560
    assert c.forecaster.next_target_attempts == c.forecaster.next_target_completed == 42
    assert out.report["forecast_substeps"] == 560
    assert not out.report["teacher_qualified"]
    for item in out.candidates:
        assert not item.score["new_predicted_foot_contact_required"]
        assert len(item.forecast.next_target_records) == len(item.forecast.next_neural_records) == 3
        np.testing.assert_array_equal(
            item.forecast.arrays["requested_offset_nm"][:30], np.tile(item.offset_nm, (30, 1))
        )
        assert not np.any(item.forecast.arrays["requested_offset_nm"][30:])
