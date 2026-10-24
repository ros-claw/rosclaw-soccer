import numpy as np
import pytest
import torch
from test_shared_native_receiving_player import fixture as motor_fixture

from rosclaw_soccer.rsi.receiving_command_actor import ReceivingCommandActor, command_inputs
from rosclaw_soccer.rsi.shared_native_receiving_command_player import (
    SharedNativeReceivingCommandPlayer,
)


def fixture(monkeypatch):
    """Actual tiny MuJoCo clock, MOCK velocity NN. Not G1 locomotion evidence."""
    w, body, _ = motor_fixture(monkeypatch)
    actor = ReceivingCommandActor(torch.zeros(112), torch.ones(112)).eval().requires_grad_(False)
    home, forward = np.array([3.0, 0.0]), np.array([1.0, 0.0])
    p = SharedNativeReceivingCommandPlayer(body, actor, home_xy=home, pass_direction=forward)
    return w, p


def base(p, anchor):
    body = p.motor
    q, v = body.layout.canonical(body.world.data)
    x, command, _, _ = command_inputs(
        q,
        v,
        body.layout.angular_velocity_body(body.world.data),
        params=body.params,
        last_action=body.last_action,
        phase=body.phase,
        home_xy=p.home_xy,
        pass_direction=p.pass_direction,
        anchor_world=anchor,
        launch_seen=p.launch_seen,
        foot_contact_seen=body.foot_contact_seen,
        other_body_contact_seen=body.other_body_contact_seen,
    )
    return x, command


def test_role_queries_actual_own_state_and_motor_recomputes_actual_command_observation(monkeypatch):
    w, p = fixture(monkeypatch)
    anchor = np.array([3.25, 0.12, 0.11])
    for step in range(4):
        expected, command = base(p, anchor)
        phase = p.motor.phase
        target, trace = p.propose(command, anchor, frame=step)
        np.testing.assert_array_equal(trace["command_inputs112"], expected)
        np.testing.assert_array_equal(trace["command_baseline3"], command)
        np.testing.assert_array_equal(trace["obs"][6:9], trace["command_policy_action3"])
        assert p.motor.phase == phase and p.motor.decisions == step
        w.advance({"a/": np.zeros(29), "b/": target})
        p.commit()
        assert p.pending_launch is None and p.motor.decisions == step + 1
    assert w.native_steps == 40
    assert p.neural_command_calls == 4 and p.motor.reference_neural_calls == 4
    assert p.motor.whole_mlp_neural_calls == 8


@pytest.mark.parametrize("fault", ["teacher_input", "premature", "double", "feedback"])
def test_no_unexecuted_feedback_or_teacher_command_repair(monkeypatch, fault):
    w, p = fixture(monkeypatch)
    anchor = np.array([3.25, 0.12, 0.11])
    _, command = base(p, anchor)
    if fault == "teacher_input":
        with pytest.raises(ValueError):
            p.propose(command + 0.1, anchor, frame=0)
        assert p.neural_command_calls == 0 and w.control_steps == 0
    else:
        target, _ = p.propose(command, anchor, frame=0)
        with pytest.raises(ValueError):
            if fault == "premature":
                p.commit()
            elif fault == "double":
                p.propose(command, anchor, frame=0)
            else:
                w.advance({"a/": np.zeros(29), "b/": target})
                p.commit(executed_action=np.full(29, 0.01, np.float32))
    assert p.faulted
    with pytest.raises(ValueError, match="latched"):
        p.propose(command, anchor, frame=0)


def test_command_protocol_not_legacy_PPO_and_counts_extra_real_network():
    from rosclaw_soccer.rsi.refinement_actor_clock import validate_actor_clock
    from rosclaw_soccer.rsi.shared_native_receiving_command_player import command_metadata

    r = command_metadata(389)
    assert r["receiver_command_neural_calls"] == 389
    assert r["receiver_command_teacher_actions_executed"] is False
    with pytest.raises(ValueError):
        validate_actor_clock(r, 389)


@pytest.mark.parametrize("replay", [False, True])
def test_command_actor_requires_explicit_qualified_motor_before_allocation(replay):
    from scripts.rsi_pass_transfer_replay_trial import replay_trial
    from scripts.rsi_pass_transfer_trial import collect_trial

    actor = ReceivingCommandActor(torch.zeros(112), torch.ones(112)).eval().requires_grad_(False)
    kwargs = dict(receiver_command_actor=actor, output=None, index=0, counts=None)
    with pytest.raises(ValueError, match="explicit qualified"):
        if replay:
            replay_trial(
                None,
                None,
                dict(assisted=False),
                dict(receiver_command_neural=True),
                raw_root=None,
                **kwargs,
            )
        else:
            collect_trial(None, None, dict(assisted=False), **kwargs)


def test_task_conditioned_command_uses_explicit_own_frame_goal_without_changing_parent_initially(
    monkeypatch,
):
    from rosclaw_soccer.rsi.receiving_task_command_actor import (
        ReceivingTaskCommandActor,
        receiving_task_context6,
    )

    w, p = fixture(monkeypatch)
    goal = np.array([3.0, 0.0, 0.11])
    anchor = np.array([3.25, 0.12, 0.11])
    child = (
        ReceivingTaskCommandActor(p.actor, torch.zeros(6), torch.ones(6))
        .eval()
        .requires_grad_(False)
    )
    player = SharedNativeReceivingCommandPlayer(
        p.motor, child, home_xy=p.home_xy, pass_direction=p.pass_direction, capture_goal_world=goal
    )
    features, command = base(p, anchor)
    q, _ = p.motor.layout.canonical(w.data)
    target, trace = player.propose(command, anchor, frame=0)
    np.testing.assert_array_equal(trace["command_inputs118"][:112], features)
    np.testing.assert_array_equal(
        trace["command_task_context6"], receiving_task_context6(q[:7], goal, p.pass_direction)
    )
    with torch.no_grad():
        expected = p.actor(torch.from_numpy(features[None]))[0].numpy()
    np.testing.assert_array_equal(trace["command_policy_action3"], expected)
    w.advance({"a/": np.zeros(29), "b/": target})
    player.commit()
    assert player.motor.decisions == 1
