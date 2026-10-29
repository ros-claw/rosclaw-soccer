import copy
from dataclasses import replace

import numpy as np
import pytest
import torch
from test_continuous_velocity_stream import fake_policy
from test_shared_native_motor_world import world
from test_velocity_prior_actor import tensors

from rosclaw_soccer.providers.g1.continuous_velocity_stream import ContinuousVelocityStream
from rosclaw_soccer.providers.g1.velocity_policy import target29
from rosclaw_soccer.rsi.anchored_receiving_motor import AnchoredReceivingMotor
from rosclaw_soccer.rsi.relay_receiver_actor_selection import select_relay_receiver_actor
from rosclaw_soccer.rsi.shared_native_receiving_player import SharedNativeReceivingPlayer
from rosclaw_soccer.rsi.velocity_prior_actor import VelocityPriorActor


def fixture(monkeypatch):
    """Actual tiny MuJoCo clock + MOCK NN/codec; NOT G1 locomotion evidence."""

    def loader(*a, **k):
        p = fake_policy()
        p.params = replace(p.params, stiffness=(20.0,) * 29, damping=(1.0,) * 29)
        return p

    monkeypatch.setattr(
        "rosclaw_soccer.rsi.shared_native_receiving_player.G1VelocityPolicy", loader
    )
    monkeypatch.setattr(
        "rosclaw_soccer.providers.g1.continuous_velocity_stream.G1VelocityPolicy", loader
    )
    w = world()
    actor = AnchoredReceivingMotor(
        VelocityPriorActor(tensors(), reference_features=14), torch.full((29,), 0.25)
    )
    actor.eval().requires_grad_(False)
    p = SharedNativeReceivingPlayer(
        w,
        "b/",
        actor,
        model=None,
        config=None,
        expected_model_hash="mock",
        expected_config_hash="mock",
    )
    old = ContinuousVelocityStream(
        None, None, expected_model_hash="mock", expected_config_hash="mock", horizon_controls=400
    )
    return w, p, old


def test_zero_identity_preserves_legacy_codec_phase_and_three_own_queries(monkeypatch):
    w, p, old = fixture(monkeypatch)
    command = np.array([0.3, 0.1, 0.2], np.float32)
    anchor = np.array([3.0, 0.0, 0.11])
    for frame in range(4):
        q, v = w.layouts["b/"].canonical(w.data)
        expected = old.propose(
            q, v, w.layouts["b/"].angular_velocity_body(w.data), command, frame=frame
        )
        target, trace = p.propose(command, anchor, frame=frame)
        np.testing.assert_array_equal(target, expected)
        np.testing.assert_array_equal(trace["obs"], old.last_observation)
        np.testing.assert_array_equal(trace["action"], old.last_action)
        assert trace["phase"] == old.phase
        assert p.decisions == frame
        w.advance({"a/": np.zeros(29), "b/": target})
        p.commit()
        assert p.decisions == frame + 1 and p.phase == old.phase
        assert p.reference_neural_calls == frame + 1
        assert p.whole_mlp_neural_calls == 2 * (frame + 1)
        assert p.attempted_neural_calls == 3 * (frame + 1)
    assert w.native_steps == 40


def test_selected_candidate_changes_actual_simulated_target(monkeypatch):
    w, parent_player, _ = fixture(monkeypatch)
    parent = parent_player.actor
    candidate = copy.deepcopy(parent)
    with torch.no_grad():
        list(candidate.plastic.parameters())[-1].add_(0.02)
    selected = select_relay_receiver_actor(
        {"receiver_learned": True, "receiver_child_candidate": True},
        whole=parent,
        parent=parent,
        candidate=candidate,
    )
    candidate_player = SharedNativeReceivingPlayer(
        w,
        "b/",
        selected,
        model=None,
        config=None,
        expected_model_hash="mock",
        expected_config_hash="mock",
    )
    cmd, anchor = np.zeros(3, np.float32), np.array([3.0, 0.0, 0.11])
    parent_target, _ = parent_player.propose(cmd, anchor, frame=0)
    candidate_target, trace = candidate_player.propose(cmd, anchor, frame=0)
    assert candidate_player.actor is candidate
    assert not np.array_equal(parent_target, candidate_target)
    w.advance({"a/": np.zeros(29), "b/": candidate_target})
    candidate_player.commit()
    np.testing.assert_array_equal(candidate_player.last_action, trace["action"])
    np.testing.assert_array_equal(w.last_applied_target("b/"), candidate_target)


def test_only_actual_executed_action_is_feedback_and_does_not_alias_caller(monkeypatch):
    w, p, _ = fixture(monkeypatch)
    cmd = np.zeros(3, np.float32)
    anchor = np.array([3.0, 0.0, 0.11])
    _, trace = p.propose(cmd, anchor, frame=0)
    actual = np.linspace(-0.01, 0.01, 29, dtype=np.float32)
    assert not np.array_equal(actual, trace["action"])
    w.advance({"a/": np.zeros(29), "b/": target29(actual, p.params)})
    p.commit(executed_action=actual)
    expected = actual.copy()
    actual.fill(10)
    _, following = p.propose(cmd, anchor, frame=1)
    np.testing.assert_array_equal(following["obs"][69:], expected)
    np.testing.assert_array_equal(p.last_action, expected)


@pytest.mark.parametrize(
    "fault",
    ["double", "premature_commit", "wrong_feedback", "too_large", "bad_clock", "gain_change"],
)
def test_wrong_clock_or_unexecuted_feedback_latches_instead_of_silent_repair(monkeypatch, fault):
    w, p, _ = fixture(monkeypatch)
    cmd = np.zeros(3, np.float32)
    anchor = np.array([3.0, 0.0, 0.11])
    if fault == "bad_clock":
        with pytest.raises(ValueError):
            p.propose(cmd, anchor, frame=1)
    elif fault == "gain_change":
        from rosclaw_soccer.rsi.shared_native_motor_world import NativePDParameters

        w.parameters["b/"] = NativePDParameters(
            np.full(29, 21.0), np.ones(29), np.full(29, 25.0)
        ).owned_copy()
        with pytest.raises(ValueError):
            p.propose(cmd, anchor, frame=0)
    else:
        target, _ = p.propose(cmd, anchor, frame=0)
        if fault == "double":
            with pytest.raises(ValueError):
                p.propose(cmd, anchor, frame=0)
        elif fault == "premature_commit":
            with pytest.raises(ValueError):
                p.commit()
        else:
            w.advance({"a/": np.zeros(29), "b/": target})
            feedback = np.full(29, 2.0 if fault == "too_large" else 0.01, np.float32)
            with pytest.raises(ValueError):
                p.commit(executed_action=feedback)
    assert p.faulted
    with pytest.raises(ValueError, match="latched"):
        p.propose(cmd, anchor, frame=0)


def test_past_contact_flags_do_not_appear_before_actual_commit(monkeypatch):
    w, p, _ = fixture(monkeypatch)
    cmd = np.zeros(3, np.float32)
    anchor = np.array([3.0, 0.0, 0.11])
    target, trace = p.propose(cmd, anchor, frame=0)
    np.testing.assert_array_equal(trace["context14"][12:], [0.0, 0.0])
    w.advance({"a/": np.zeros(29), "b/": target})
    # Synthetic sensor witness tests causality only, not a claimed physical contact.
    w.last_ball_contact_bodies["b/"] = {"b/left_ankle_roll_link": 2.0, "b/left_knee_link": 3.0}
    p.commit()
    _, following = p.propose(cmd, anchor, frame=1)
    np.testing.assert_array_equal(following["context14"][12:], [1.0, 1.0])
