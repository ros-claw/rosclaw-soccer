from types import SimpleNamespace

import numpy as np
import pytest
from test_dual_recurrent_motor import model
from test_shared_native_motor_world import world

from rosclaw_soccer.rsi.shared_native_neural_player import SharedNativeNeuralPlayer


def players():
    w = world()
    _, actor = model()
    actor.requires_grad_(False)
    clip = SimpleNamespace(
        length=1,
        joint_pos=np.full((1, 29), 0.01, np.float32),
        joint_vel=np.zeros((1, 29), np.float32),
        body_ang_vel_w=np.zeros((1, 1, 3), np.float32),
    )

    def observe(view, *args, **kwargs):
        obs = np.zeros(160, np.float32)
        obs[:29] = view.joint_pos
        return obs

    return w, {
        prefix: SharedNativeNeuralPlayer(
            w,
            prefix,
            actor,
            clip=clip,
            torso_index=0,
            build_observation=observe,
            goal=np.array([6.0, 0.0, 0.11]),
            offset=np.zeros(29),
            scale=np.full(29, 0.5),
            order=np.arange(29),
        )
        for prefix in w.layouts
    }


def test_private_memory_and_two_phase_commit_prevent_double_proposals():
    w, ps = players()
    a, b = ps["a/"], ps["b/"]
    assert a.state[0].data_ptr() != b.state[0].data_ptr()
    targets, traces = {}, {}
    for prefix, p in ps.items():
        targets[prefix], traces[prefix] = p.propose()
        with pytest.raises(ValueError, match="one own proposal"):
            p.propose()
        with pytest.raises(ValueError, match="executed shared"):
            p.commit()
        assert p.state[0].count_nonzero() == 0
    w.advance(targets)
    for prefix, p in ps.items():
        p.commit()
        np.testing.assert_array_equal(p.state[0].numpy(), traces[prefix]["NN_h_after"])
        assert p.decisions == 1 and p.phase == "MOTION_COMPLETE"
        with pytest.raises(ValueError):
            p.commit()
        before = tuple(t.clone() for t in p.state)
        p.begin_standing()
        assert all(x.equal(y) for x, y in zip(before, p.state, strict=True))
    targets = {prefix: p.propose()[0] for prefix, p in ps.items()}
    w.advance(targets)
    for p in ps.values():
        p.commit()
    assert w.native_steps == 20


def test_reference_end_cannot_be_silently_padded():
    w, ps = players()
    proposals = {prefix: p.propose()[0] for prefix, p in ps.items()}
    w.advance(proposals)
    for p in ps.values():
        p.commit()
        with pytest.raises(ValueError):
            p.propose()


def test_committed_options_keep_private_state_through_actual_commit_and_standing():
    from test_committed_neural_motor import fixture

    from rosclaw_soccer.rsi.private_motor_history import clone_history

    w, old = players()
    _, _, actor, obs_a, obs_b = fixture()
    ps = {}
    for prefix, observation in zip(w.layouts, (obs_a, obs_b), strict=True):
        p = old[prefix]

        def observe(*args, value=observation, **kwargs):
            return value[0, 0].numpy().copy()

        ps[prefix] = SharedNativeNeuralPlayer(
            w,
            prefix,
            actor,
            clip=p.clip,
            torso_index=0,
            build_observation=observe,
            goal=p.goal,
            offset=p.offset,
            scale=p.scale,
            order=p.order,
        )
    for control in range(2):
        targets, traces = {}, {}
        for prefix, p in ps.items():
            targets[prefix], traces[prefix] = p.propose()
            assert p.state.controls == control  # no commit before actual integration
            assert traces[prefix]["NN_option_controls_after"] == control + 1
        w.advance(targets)
        for prefix, p in ps.items():
            p.commit()
            assert p.state.option == (1 if prefix == "a/" else 0)
            assert p.state.controls == control + 1
            if control == 0:
                before = clone_history(p.state)
                p.begin_standing()
                assert p.state.option == before.option
                assert p.state.generation_hash == before.generation_hash
    assert ps["a/"].state.recurrent[0].data_ptr() != ps["b/"].state.recurrent[0].data_ptr()


def test_committed_player_rejects_cross_generation_handoff_before_physics():
    from dataclasses import replace

    from test_committed_neural_motor import fixture

    w, ps = players()
    _, _, actor, _, _ = fixture()
    p = ps["a/"]
    state = replace(actor.initial_state(), generation_hash="sha256:" + "b" * 64)
    with pytest.raises(ValueError, match="generation"):
        SharedNativeNeuralPlayer(
            w,
            "a/",
            actor,
            clip=p.clip,
            torso_index=0,
            build_observation=p.build_observation,
            goal=p.goal,
            offset=p.offset,
            scale=p.scale,
            order=p.order,
            initial_state=state,
        )
    assert w.native_steps == 0


def test_standing_requires_actual_reference_boundary():
    _, ps = players()
    with pytest.raises(ValueError, match="complete motion"):
        ps["a/"].begin_standing()


def test_default_commit_rejects_unexecuted_nominal_without_mutating_history():
    w, ps = players()
    targets = {prefix: p.propose()[0] for prefix, p in ps.items()}
    targets["a/"] = targets["a/"] + 0.01
    w.advance(targets)
    a = ps["a/"]
    before = tuple(t.clone() for t in a.state)
    with pytest.raises(ValueError, match="ACTUAL executed target"):
        a.commit()
    assert a.decisions == 0 and a.pending is not None
    assert all(x.equal(y) for x, y in zip(before, a.state, strict=True))


def test_explicit_teacher_requires_own_reference_evidence():
    w, ps = players()
    targets, traces = {}, {}
    for prefix, p in ps.items():
        targets[prefix], traces[prefix] = p.propose()
    w.advance(targets)
    with pytest.raises(ValueError, match="own reference"):
        ps["a/"].commit(executed_action=traces["a/"]["NN_action"])


def test_teacher_commits_actual_action_not_nominal_and_copies_feedback():
    w, ps = players()
    a = ps["a/"]
    _, trace = a.propose(with_reference_evidence=True)
    action = (trace["NN_reference_nominal"] + np.float32(0.2)).astype(np.float32)
    target = a.offset + action[a.order] * a.scale
    btarget, _ = ps["b/"].propose()
    w.advance({"a/": target, "b/": btarget})
    a.commit(executed_action=action)
    np.testing.assert_array_equal(a.last_action, action)
    np.testing.assert_array_equal(a.state[0].numpy(), trace["NN_h_after"])
    assert not np.array_equal(a.last_action, trace["NN_action"])
    action[:] = 9
    assert not np.array_equal(a.last_action, action)


def test_teacher_authority_and_feedback_shape_are_fail_closed():
    w, ps = players()
    a = ps["a/"]
    target, trace = a.propose(with_reference_evidence=True)
    btarget, _ = ps["b/"].propose()
    w.advance({"a/": target, "b/": btarget})
    for bad in (
        np.zeros(28, np.float32),
        np.zeros(29, np.float64),
        np.full(29, np.nan, np.float32),
    ):
        with pytest.raises(ValueError, match="finite actual"):
            a.commit(executed_action=bad)
    with pytest.raises(ValueError, match="motor authority"):
        a.commit(executed_action=trace["NN_reference_nominal"] + np.float32(0.5))
    assert a.decisions == 0
    a.commit(executed_action=trace["NN_action"])


def test_reference_evidence_mode_preserves_single_forward_action_and_history():
    _, ps = players()
    ordinary, evidence = ps["a/"], ps["b/"]
    # Compare the same measured body; separate private histories remain intact.
    evidence.view = ordinary.view
    t0, r0 = ordinary.propose()
    t1, r1 = evidence.propose(with_reference_evidence=True)
    np.testing.assert_array_equal(t0, t1)
    assert "NN_reference_nominal" not in r0
    for key, value in r0.items():
        np.testing.assert_array_equal(value, r1[key])
