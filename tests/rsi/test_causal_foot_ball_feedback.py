from types import SimpleNamespace

import numpy as np
import pytest

from rosclaw_soccer.rsi.causal_foot_ball_feedback import (
    CausalFootBallFeedback,
    closest_approach,
    damped_joint_delta,
    feedback_parameters,
    task_displacement,
)


def test_current_state_correction_brakes_and_aligns_without_vertical_change():
    point, ball = np.array([0.0, 0.02, 0.1]), np.array([0.12, 0.0, 0.11])
    pv, bv, goal = np.array([4.0, 0.2, 0.0]), np.zeros(3), np.array([4.0, 0.0, 0.11])
    before = [a.copy() for a in (point, ball, pv, bv, goal)]
    task, measured = task_displacement(point, ball, pv, bv, goal, variant=2)
    assert task[0] < 0 and task[1] < 0 and task[2] == 0
    assert measured[0] == 4.0
    assert np.linalg.norm(task) <= 0.02500000000001
    for a, b in zip(before, (point, ball, pv, bv, goal), strict=True):
        np.testing.assert_array_equal(a, b)


def test_relative_velocity_and_planar_translation_invariance():
    args = [
        np.array(a, dtype=np.float64)
        for a in ([0, 0.02, 0.1], [0.12, 0, 0.11], [4, 0.2, 0], [0, 0, 0], [4, 0, 0.11])
    ]
    expected = task_displacement(*args, variant=1)
    translated = [a.copy() for a in args]
    for i in (0, 1, 4):
        translated[i] += np.array([2, -3, 0])
    for i in (2, 3):
        translated[i] += np.array([1, -2, 0.3])
    actual = task_displacement(*translated, variant=1)
    for a, b in zip(actual, expected, strict=True):
        np.testing.assert_allclose(a, b, atol=1e-15)


def test_slow_leg_is_not_accelerated_into_ball():
    task, _ = task_displacement(
        np.zeros(3),
        np.array([0.12, 0, 0]),
        np.array([2.0, 0, 0]),
        np.zeros(3),
        np.array([4.0, 0, 0]),
        variant=3,
    )
    np.testing.assert_array_equal(task, np.zeros(3))


def test_damped_jacobian_handles_singular_and_bounded_deltas():
    np.testing.assert_array_equal(damped_joint_delta(np.zeros((3, 6)), np.ones(3)), np.zeros(6))
    jac = np.concatenate([np.eye(3), np.zeros((3, 3))], axis=1)
    delta = damped_joint_delta(jac, np.array([1.0, -1.0, 0.1]))
    assert delta[0] == 0.12 and delta[1] == -0.12
    assert (np.abs(delta) <= 0.12).all()


@pytest.mark.parametrize("variant", [True, -1, 4, 1.0])
def test_bad_config_rejected(variant):
    with pytest.raises(ValueError):
        feedback_parameters(variant)


@pytest.mark.parametrize("fault", ["dtype", "nan", "shape", "target"])
def test_invalid_current_measurements_rejected(fault):
    args = [np.zeros(3) for _ in range(5)]
    args[-1][0] = 4
    if fault == "dtype":
        args[0] = args[0].astype(np.float32)
    elif fault == "nan":
        args[2][0] = np.nan
    elif fault == "shape":
        args[3] = np.zeros(2)
    else:
        args[-1] *= 0
    with pytest.raises(ValueError):
        task_displacement(*args, variant=1)


def measured_adapter(variant=2):
    import mujoco
    from test_shared_native_motor_world import source_xml

    from rosclaw_soccer.rsi.shared_native_motor_world import (
        NativePDParameters,
        SharedNativeMotorWorld,
    )
    from rosclaw_soccer.rsi.shared_native_scene import shared_native_scene

    xml = source_xml().replace('name="link5"', 'name="left_ankle_roll_link"')
    model = mujoco.MjModel.from_xml_string(shared_native_scene(xml, prefixes=("a/", "b/")))
    data = mujoco.MjData(model)
    params = NativePDParameters(np.full(29, 20.0), np.ones(29), np.full(29, 25.0))
    w = SharedNativeMotorWorld(
        model, data, players={"a/": params, "b/": params}, ball_body_name="ball"
    )
    mujoco.mj_forward(model, data)
    foot = model.body("a/left_ankle_roll_link").id
    point = data.xpos[foot] + np.array(feedback_parameters(variant)["point_body_local_m"])
    al = w.layouts["a/"]
    data.qpos[al.bq : al.bq + 3] = point + np.array([0.12, 0.02, 0.01])
    data.qvel[al.rv : al.rv + 3] = [4.0, 0.5, 0]
    mujoco.mj_forward(model, data)
    ep = SimpleNamespace(order=np.arange(29)[::-1], scale=np.full(29, 0.5, np.float32))
    goal = point + np.array([4.0, 0.02, 0.01])
    return w, ep, CausalFootBallFeedback(w, ep, goal=goal, variant=variant)


def test_actual_mujoco_measurement_does_not_step_or_mutate_and_keeps_total_authority(monkeypatch):
    import mujoco

    w, ep, adapter = measured_adapter()
    before = [getattr(w.data, k).copy() for k in ("qpos", "qvel", "ctrl")]
    monkeypatch.setattr(mujoco, "mj_step", lambda *args: pytest.fail("feedback must not integrate"))
    proposal, reference = np.full(29, 0.39, np.float32), np.zeros(29, np.float32)
    action, delta, _, features = adapter.apply(proposal, reference, frame=100, standing=False)
    assert features.shape == (24,) and np.isfinite(features).all()
    assert features[18] > 0 and np.linalg.norm(delta) > 0
    assert np.max(np.abs((action - reference)[ep.order] * ep.scale)) <= 0.20002
    changed = np.zeros(29, dtype=bool)
    changed[ep.order] = delta != 0
    np.testing.assert_array_equal(action[~changed], proposal[~changed])
    for a, name in zip(before, ("qpos", "qvel", "ctrl"), strict=True):
        np.testing.assert_array_equal(a, getattr(w.data, name))
    assert w.native_steps == 0 and w.data.time == 0


@pytest.mark.parametrize(
    "frame,standing,variant", [(87, False, 1), (120, False, 1), (100, True, 1), (100, False, 0)]
)
def test_no_feedback_outside_window_or_zero_variant(frame, standing, variant):
    _, _, adapter = measured_adapter(variant)
    p = np.full(29, 0.1, np.float32)
    actual, delta, clipped, features = adapter.apply(
        p, np.zeros(29, np.float32), frame=frame, standing=standing
    )
    np.testing.assert_array_equal(actual, p)
    assert not delta.any() and not features.any() and not clipped


@pytest.mark.parametrize("collector", ["rsi_pass_transfer_trial", "rsi_pass_transfer_replay_trial"])
def test_feedback_rejects_stacked_teachers_before_prepare(collector):
    import importlib

    module = importlib.import_module("scripts." + collector)
    setup = SimpleNamespace(prepare=lambda *args: pytest.fail("invalid feedback must not allocate"))
    common = dict(output=None, index=0, counts={}, feedback_variant=1)
    if collector.endswith("replay_trial"):
        with pytest.raises(ValueError):
            module.replay_trial(
                setup, None, dict(assisted=True), dict(feedback_variant=1), raw_root=None, **common
            )
    else:
        with pytest.raises(ValueError):
            module.collect_trial(setup, None, dict(assisted=True), **common)


@pytest.mark.parametrize(
    "position,velocity,expected_time,expected_distance",
    [
        ([0.6, 0, 0], [-4.0, 0, 0], 0.12, 0.12),
        ([0.2, 0, 0], [-4.0, 0, 0], 0.05, 0.0),
        ([0.6, 0, 0], [4.0, 0, 0], 0.0, 0.6),
        ([0.6, 0, 0], [0.0, 0, 0], 0.0, 0.6),
        ([0.6, 0.4, 0], [-4.0, 0, 0], 0.12, float(np.hypot(0.12, 0.4))),
    ],
)
def test_bounded_current_velocity_closest_approach(
    position, velocity, expected_time, expected_distance
):
    p, v = np.array(position, dtype=np.float64), np.array(velocity, dtype=np.float64)
    p0, v0 = p.copy(), v.copy()
    distance, time = closest_approach(p, v, seconds=0.12)
    assert time == pytest.approx(expected_time)
    assert distance == pytest.approx(expected_distance)
    np.testing.assert_array_equal(p, p0)
    np.testing.assert_array_equal(v, v0)


@pytest.mark.parametrize("seconds", [True, 0.1, 0.5, float("nan")])
def test_undeclared_anticipation_is_not_a_tunable_runtime_horizon(seconds):
    with pytest.raises(ValueError):
        closest_approach(np.ones(3), -np.ones(3), seconds=seconds)
    with pytest.raises(ValueError):
        feedback_parameters(1, anticipation_seconds=seconds)


def test_anticipatory_gate_acts_before_instantaneous_distance_gate_without_stepping():
    import mujoco

    w, ep, original = measured_adapter(2)
    foot = original.foot
    point = w.data.xpos[foot] + w.data.xmat[foot].reshape(3, 3) @ np.array(
        original.cfg["point_body_local_m"]
    )
    w.data.qpos[original.layout.bq : original.layout.bq + 3] = point + np.array([0.6, 0.02, 0.01])
    mujoco.mj_forward(w.model, w.data)
    future = CausalFootBallFeedback(w, ep, goal=original.goal, variant=2, anticipation_seconds=0.12)
    proposal = np.zeros(29, np.float32)
    old = original.apply(proposal, proposal, frame=100, standing=False)
    new = future.apply(proposal, proposal, frame=100, standing=False)
    assert old[3][18] == 0 and not old[1].any()
    assert new[3].shape == (26,) and new[3][18] > 0 and new[1].any()
    assert 0 <= new[3][25] <= 0.12
    assert new[3][24] < 0.35
    assert w.native_steps == 0 and w.data.time == 0


def test_default_feedback_declaration_remains_original_without_anticipation():
    assert feedback_parameters(2) == feedback_parameters(2, anticipation_seconds=0.0)
    assert "anticipation_seconds" not in feedback_parameters(2)
