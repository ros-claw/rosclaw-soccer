import copy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from test_shared_native_neural_player import players

from scripts.rsi_pass_transfer_trial import collect_trial


class FixtureVelocity:
    """No ONNX claim: isolate collection/feedback plumbing in a toy world."""

    def __init__(self, *args, **kwargs):
        self.last_observation = np.zeros(98, np.float32)
        self.last_action = np.zeros(29, np.float32)
        self.phase = 0.0

    def propose(self, *args, **kwargs):
        return np.zeros(29, np.float64)


@pytest.mark.parametrize("prefix", [0, 80])
def test_actual_pair_stream_and_contiguous_expert_suffix(tmp_path, monkeypatch, prefix):
    w, ps = players()
    original = ps["a/"]
    actor = original.actor
    student = copy.deepcopy(actor)
    with torch.no_grad():
        list(student.plastic.parameters())[2].add_(0.001)
        list(student.plastic.parameters())[-1].add_(0.001)
    clip = SimpleNamespace(
        length=100,
        joint_pos=np.zeros((100, 29), np.float32),
        joint_vel=np.zeros((100, 29), np.float32),
        body_ang_vel_w=np.zeros((100, 1, 3), np.float32),
    )

    def observe(view, _unused, _frame, last, *args, **kwargs):
        obs = np.zeros(160, np.float32)
        obs[:29] = view.joint_pos
        obs[125:154] = last
        return obs

    ep = SimpleNamespace(
        clip=clip,
        torso_index=0,
        build_observation=observe,
        offset=original.offset,
        scale=original.scale,
        order=original.order,
    )
    setup = SimpleNamespace(
        prepare=lambda *a, **k: (ep, w, np.array([1.0, 0.0]), np.array([2.0, 0.0]), np.zeros(2)),
        velocity_path=Path("fixture-model"),
        velocity_config=Path("fixture-config"),
        pins={"fixture-model": "fixture", "fixture-config": "fixture"},
    )
    monkeypatch.setattr("scripts.rsi_pass_transfer_trial.ContinuousVelocityStream", FixtureVelocity)
    course = dict(
        context=1,
        side=-0.55,
        target=[6.0, 0.0, 0.11],
        assisted=True,
        center=100,
        latent=np.zeros((2, 6)).tolist(),
        foot="left",
    )
    counts = dict(native_steps=0, actor_calls=0, shadow_actor_calls=0)
    r = collect_trial(
        setup,
        actor,
        course,
        output=tmp_path,
        index=0,
        counts=counts,
        student=student,
        student_prefix_controls=prefix,
    )
    with np.load(tmp_path / r["artifacts"]["A"]["path"], allow_pickle=False) as z:
        n = len(z["NN_obs"])
        assert n == 200
        np.testing.assert_array_equal(z["student_executed"], np.arange(n) < prefix)
        np.testing.assert_array_equal(z["STUDENT_NN_obs"], z["TEACHER_NN_obs"])
        assert not np.array_equal(z["STUDENT_NN_action"], z["TEACHER_NN_action"])
        assert not np.array_equal(z["STUDENT_NN_h_after"], z["TEACHER_NN_h_after"])
        np.testing.assert_array_equal(z["NN_obs"][1:, 125:154], z["executed_action"][:-1])
        assert not z["teacher_delta_HG"][:prefix].any()
        np.testing.assert_array_equal(
            z["executed_action"][:prefix], z["STUDENT_NN_action"][:prefix]
        )
        np.testing.assert_array_equal(
            z["executed_action"][prefix:], z["teacher_advice_action"][prefix:]
        )
    assert counts == dict(native_steps=2000, actor_calls=400, shadow_actor_calls=200)
    assert r["total_neural_calls"] == 600 and r["PPO_actor_rows"] == 0
    assert not r["teacher_advice_alone_physically_qualified"]
    from scripts.rsi_pass_transfer_replay_trial import replay_trial

    monkeypatch.setattr(
        "scripts.rsi_pass_transfer_replay_trial.ContinuousVelocityStream", FixtureVelocity
    )
    audit = tmp_path / "own"
    audit.mkdir()
    # Reconstruct the whole toy initial world; the manual clock may not execute
    # the producer's saved actions or call the producer's world.advance().
    fresh, _ = players()
    setup.prepare = lambda *a, **k: (
        ep,
        fresh,
        np.array([1.0, 0.0]),
        np.array([2.0, 0.0]),
        np.zeros(2),
    )
    own_counts = dict(native_steps=0, actor_calls=0, shadow_actor_calls=0)
    own, _ = replay_trial(
        setup,
        actor,
        course,
        r,
        raw_root=tmp_path,
        output=audit,
        index=0,
        counts=own_counts,
        student=student,
        student_prefix_controls=prefix,
    )
    assert own_counts == counts
    assert own == {k: v for k, v in r.items() if k != "artifacts"}


@pytest.mark.parametrize(
    "student,prefix", [(None, 80), (object(), None), (object(), True), (object(), 81)]
)
def test_invalid_prefix_is_rejected_before_world_allocation(student, prefix, tmp_path):
    with pytest.raises(ValueError, match="prefix"):
        collect_trial(
            None,
            None,
            {},
            output=tmp_path,
            index=0,
            counts={},
            student=student,
            student_prefix_controls=prefix,
        )
    assert not list(tmp_path.iterdir())
