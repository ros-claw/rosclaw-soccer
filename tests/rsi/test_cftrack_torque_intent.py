from types import SimpleNamespace

import numpy as np
import pytest

from rosclaw_soccer.rsi.cftrack_torque_intent import (
    CFtrackTorqueIntentPlan,
    CFtrackTorqueIntentPolicy,
)


def policy(value=0.2, start=0, peak=24):
    delta = [0.0] * 29
    delta[7] = value
    return CFtrackTorqueIntentPolicy(
        CFtrackTorqueIntentPlan(
            start, peak, tuple(delta), "sha256:" + "a" * 64, "sha256:" + "b" * 64
        ),
        np.full(29, 10.0),
    )


def frame(q=0.0):
    return SimpleNamespace(
        joint_position=np.full(29, q),
        joint_lower_limits=np.full(29, -1.0),
        joint_upper_limits=np.full(29, 1.0),
    )


def test_zero_intent_preserves_signed_parent_and_requires_exact_feedback():
    p = policy(0.0)
    parent = np.full(29, -0.0)
    for _ in range(500):
        x = p.command(frame(), parent)
        np.testing.assert_array_equal(x.view(np.uint64), parent.view(np.uint64))
        p.note_applied(x.copy())
    assert p.steps == 500 and p.saturation_steps == p.stop_guard_steps == 0


def test_effort_caps_slew_and_measured_stop_are_explicit():
    p = policy()
    outputs = []
    for _ in range(481):
        x = p.command(frame(), np.zeros(29))
        outputs.append(x[7])
        p.note_applied(x)
    assert max(outputs) == pytest.approx(2.0)
    assert max(abs(np.diff(outputs))) / 0.002 <= 20.0 + 1e-10
    p = policy()
    for _ in range(241):
        x = p.command(frame(1.0), np.full(29, 9.0))
        p.note_applied(x)
    assert x[7] == 9.0 and p.stop_guard_steps > 0
    p = policy()
    for _ in range(241):
        x = p.command(frame(), np.full(29, 9.0))
        p.note_applied(x)
    assert x[7] == 10.0 and p.saturation_steps > 0


def test_out_of_range_pd_setpoint_is_not_an_input_to_torque_protocol():
    p = policy()
    p.steps = 240
    x = p.command(frame(), np.zeros(29))
    assert x[7] == 2.0
    with pytest.raises(ValueError, match="consecutive"):
        p.command(frame(), np.zeros(29))
    with pytest.raises(ValueError, match="feedback"):
        p.note_applied(x + 0.01)
    assert p.failed


@pytest.mark.parametrize("value,peak", [(float("nan"), 24), (0.251, 24), (0.25, 1)])
def test_invalid_envelope_rejected(value, peak):
    with pytest.raises(ValueError):
        policy(value, peak=peak)


def test_native_layer_accepts_only_bound_plan_and_zero_is_bit_exact():
    from test_shared_native_torque_layer import world

    from rosclaw_soccer.rsi.shared_native_torque_layer import SharedNativeTorqueLayer
    from rosclaw_soccer.sim.contracts import hash_json

    actual, reference = world(), world()
    plan = CFtrackTorqueIntentPlan(
        0, 24, (0.0,) * 29, hash_json(actual.layouts["a/"].profile()), "sha256:" + "b" * 64
    )
    p = CFtrackTorqueIntentPolicy(plan, actual.parameters["a/"].effort)
    layer = SharedNativeTorqueLayer(
        actual,
        "a/",
        p,
        target_y_m=0.0,
        target_z_m=0.11,
        horizon_controls=2,
        expected_parent_policy_hash=plan.parent_policy_hash,
    )
    targets = {k: np.zeros(29) for k in actual.layouts}
    for _ in range(2):
        old = reference.advance(targets)
        new = actual.advance(targets, torque_layers={"a/": layer})
        for key in ("qpos", "qvel", "torque", "ball_forces"):
            np.testing.assert_array_equal(old[key], new[key])
    assert p.steps == 20
    with pytest.raises(ValueError, match="reset"):
        p.reset()


@pytest.mark.parametrize("fault", ["body", "parent", "caps"])
def test_native_layer_rejects_wrong_binding_before_physics(fault):
    from test_shared_native_torque_layer import world

    from rosclaw_soccer.rsi.shared_native_torque_layer import SharedNativeTorqueLayer
    from rosclaw_soccer.sim.contracts import hash_json

    w = world()
    parent = "sha256:" + "b" * 64
    plan = CFtrackTorqueIntentPlan(
        0,
        24,
        (0.0,) * 29,
        "sha256:" + "a" * 64 if fault == "body" else hash_json(w.layouts["a/"].profile()),
        "sha256:" + "a" * 64 if fault == "parent" else parent,
    )
    effort = w.parameters["a/"].effort.copy()
    if fault == "caps":
        effort[0] += 0.1
    p = CFtrackTorqueIntentPolicy(plan, effort)
    with pytest.raises(ValueError, match="actual body"):
        SharedNativeTorqueLayer(
            w,
            "a/",
            p,
            target_y_m=0.0,
            target_z_m=0.11,
            horizon_controls=2,
            expected_parent_policy_hash=parent,
        )
    assert not w._torque_layer_owners and w.native_steps == 0
