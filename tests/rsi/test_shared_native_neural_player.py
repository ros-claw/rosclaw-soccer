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
