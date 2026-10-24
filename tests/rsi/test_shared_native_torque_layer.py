from types import SimpleNamespace

import numpy as np
import pytest
from rosclaw.simforge.g1_neural_torque import (
    G1_NEURAL_TORQUE_OBSERVATIONS,
    G1NeuralTorquePolicy,
    G1TeacherTorqueCollector,
)
from rosclaw.simforge.tasks.g1_goalforge.concepts import G1_HARD_TORQUE_LIMITS
from test_shared_native_motor_world import source_xml

from rosclaw_soccer.rsi.shared_native_motor_world import NativePDParameters, SharedNativeMotorWorld
from rosclaw_soccer.rsi.shared_native_scene import shared_native_scene
from rosclaw_soccer.rsi.shared_native_torque_layer import SharedNativeTorqueLayer


def world():
    import mujoco

    xml = (
        source_xml()
        .replace('name="link0"', 'name="torso_link"')
        .replace('name="link1"', 'name="left_ankle_roll_link"')
        .replace('name="link2"', 'name="right_ankle_roll_link"')
    )
    m = mujoco.MjModel.from_xml_string(shared_native_scene(xml, prefixes=("a/", "b/")))
    d = mujoco.MjData(m)
    d.qpos[m.joint("b/root").qposadr[0]] = 3.0
    mujoco.mj_forward(m, d)
    p = NativePDParameters(np.full(29, 20.0), np.ones(29), np.full(29, 25.0))
    return SharedNativeMotorWorld(m, d, players={"a/": p, "b/": p}, ball_body_name="ball")


def layer(w, prefix="a/", policy=None, horizon=2):
    return SharedNativeTorqueLayer(
        w,
        prefix,
        policy or G1TeacherTorqueCollector(),
        target_y_m=0.0,
        target_z_m=0.11,
        horizon_controls=horizon,
    )


def test_pass_through_core_teacher_is_bit_exact_and_has_native_feedback():
    reference, actual = world(), world()
    a, b = layer(actual), layer(actual, "b/")
    targets = {"a/": np.linspace(-0.01, 0.01, 29), "b/": np.linspace(0.01, 0.02, 29)}
    torques = []
    for _ in range(2):
        old = reference.advance(targets)
        new = actual.advance(targets, torque_layers={"a/": a, "b/": b})
        for key in ("qpos", "qvel", "torque", "ball_forces"):
            np.testing.assert_array_equal(old[key], new[key])
        np.testing.assert_array_equal(new["parent_torque"], new["torque"])
        assert not new["applied_torque_change"].any()
        torques.append(new["torque"])
    episode = a.policy.episode()
    assert episode.observations.shape == (20, len(G1_NEURAL_TORQUE_OBSERVATIONS))
    assert a.commands == a.applied == b.commands == b.applied == 20
    truth = np.concatenate(torques)[:, actual.layouts["a/"].ai]
    np.testing.assert_array_equal(episode.actions, truth.astype(np.float32))
    np.testing.assert_array_equal(episode.actions, episode.parent_actions)
    np.testing.assert_array_equal(episode.observations[0, -29:], np.zeros(29, np.float32))
    np.testing.assert_array_equal(
        episode.observations[1:, -29:],
        (truth[:-1] / np.array(G1_HARD_TORQUE_LIMITS)).astype(np.float32),
    )


def test_registered_layer_cannot_be_silently_omitted_or_outlive_deadline():
    w = world()
    a = layer(w, horizon=1)
    targets = {p: np.zeros(29) for p in w.layouts}
    with pytest.raises(ValueError, match="all registered"):
        w.advance(targets)
    assert w.native_steps == 0
    w.advance(targets, torque_layers={"a/": a})
    with pytest.raises(ValueError, match="bound private"):
        w.advance(targets, torque_layers={"a/": a})
    assert w.native_steps == 10


def test_one_policy_cannot_own_two_bodies_or_two_worlds():
    w = world()
    a = layer(w)
    with pytest.raises(ValueError, match="private Core"):
        layer(w, "b/", a.policy)
    with pytest.raises(ValueError, match="private Core"):
        layer(world(), "a/", a.policy)
    assert a.commands == 0


@pytest.mark.parametrize("bad", ["nonfinite", "overcap", "exception"])
def test_bad_core_policy_output_latches_both_layer_and_world_without_step(bad):
    w = world()
    a = layer(w)

    def invalid(frame, parent):
        if bad == "exception":
            raise RuntimeError("deliberate fixture failure")
        result = parent.copy()
        result[0] = np.nan if bad == "nonfinite" else 26.0
        return result

    a.policy.command = invalid
    before = w.data.qpos.copy()
    with pytest.raises((ValueError, RuntimeError)):
        w.advance({p: np.zeros(29) for p in w.layouts}, torque_layers={"a/": a})
    assert w.failed and a.failed and w.native_steps == 0
    np.testing.assert_array_equal(before, w.data.qpos)


def test_unknown_policy_class_and_wrong_artifact_body_are_rejected():
    w = world()
    with pytest.raises(ValueError, match="private Core"):
        layer(w, policy=SimpleNamespace())
    # A deliberately malformed object never reaches reset, inference or physics.
    p = G1NeuralTorquePolicy.__new__(G1NeuralTorquePolicy)
    p.artifact = SimpleNamespace(body_hash="wrong", parent_policy_hash="wrong")
    with pytest.raises(ValueError, match="actual body"):
        layer(w, policy=p)
    assert w.native_steps == 0


def test_note_applied_rejects_unobserved_fabricated_feedback():
    w = world()
    a = layer(w)
    with pytest.raises(ValueError, match="actual native"):
        a.note_applied(np.zeros(29))
    assert a.failed


def test_layer_from_another_world_cannot_be_executed():
    w, other = world(), world()
    a = layer(other)
    with pytest.raises(ValueError, match="bound private"):
        w.advance({p: np.zeros(29) for p in w.layouts}, torque_layers={"a/": a})
    assert w.native_steps == other.native_steps == 0


def test_zero_intervention_executes_parent_and_records_actual_not_discarded_torque():
    w, reference = world(), world()
    policy = G1TeacherTorqueCollector()
    original = policy.command

    def biased(frame, parent):
        original(frame, parent)
        return parent + 0.1

    policy.command = biased
    a = SharedNativeTorqueLayer(
        w,
        "a/",
        policy,
        target_y_m=0.0,
        target_z_m=0.11,
        horizon_controls=1,
        intervention_fraction=0.0,
    )
    targets = {p: np.zeros(29) for p in w.layouts}
    old = reference.advance(targets)
    actual = w.advance(targets, torque_layers={"a/": a})
    np.testing.assert_array_equal(old["qpos"], actual["qpos"])
    assert not actual["applied_torque_change"].any()
    assert not policy.episode().actions.any()


def test_invalid_discarded_torque_is_not_silently_repaired_at_zero_fraction():
    w = world()
    policy = G1TeacherTorqueCollector()
    a = SharedNativeTorqueLayer(
        w,
        "a/",
        policy,
        target_y_m=0.0,
        target_z_m=0.11,
        horizon_controls=1,
        intervention_fraction=0.0,
    )
    policy.command = lambda frame, parent: np.full(29, np.nan)
    with pytest.raises(ValueError):
        w.advance({p: np.zeros(29) for p in w.layouts}, torque_layers={"a/": a})
    assert w.failed and w.native_steps == 0
