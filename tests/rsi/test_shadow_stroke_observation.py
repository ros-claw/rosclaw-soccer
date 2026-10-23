from types import SimpleNamespace

import numpy as np
import pytest
import torch


@pytest.mark.parametrize("stroke_foot", ["left", "right"])
def test_shadow_auditor_consumes_the_single_observation_without_reading_episode_again(
    monkeypatch, stroke_foot
):
    from test_same_state_motor_pair import pair

    import rosclaw_soccer.rsi.paid_run_teacher_audit as audit
    import scripts.rsi_stroke_search_common as common
    from scripts.rsi_stroke_world_audit import replay_stroke_world

    p = pair()
    observation = np.random.default_rng(5).normal(size=160).astype(np.float32)
    private = p.initial_state()
    actual, teacher, after, evidence = p.observe(
        torch.from_numpy(observation).reshape(1, 1, 160), private
    )
    goal = np.zeros(3)

    def forbidden_second_read():
        raise AssertionError("a second observation read is forbidden")

    ep = SimpleNamespace(frame=0, goal=goal, observation=forbidden_second_read)
    monkeypatch.setattr(common, "make_episode", lambda *_: ep)

    def identity_stroke(actual, mean, *_, foot):
        assert actual is ep and foot == stroke_foot
        return mean, np.zeros(29), False

    monkeypatch.setattr(common, "stroke_action", identity_stroke)
    monkeypatch.setattr(common, "score", lambda *_, **kw: ({"test": True}, None, False))
    kick = {
        "NN_obs": observation[None],
        "NN_reference_nominal": evidence["teacher"]["reference_nominal"][0].numpy(),
        "PAIR_student_mean": actual[0].numpy(),
        "PAIR_student_executed": np.array([True]),
        "STROKE_delta_HG": np.zeros((1, 29)),
        "STROKE_authority_clipped": np.array([False]),
    }
    for j, name in enumerate(("h", "c")):
        kick[f"PAIR_student_{name}_before"] = private[0][j].numpy()[None]
        kick[f"PAIR_student_{name}_after"] = after[0][j].numpy()[None]

    def one_read_replay(ep, actor, kick, stand, *, action_adapter, **kwargs):
        e = dict(evidence["teacher"], actual_observation=observation.copy())
        applied, _ = action_adapter(ep, teacher[0, 0].numpy().copy(), e)
        assert np.array_equal(applied, actual[0, 0].numpy())

    monkeypatch.setattr(audit, "replay_teacher_world", one_read_replay)
    counts = dict(neural=0, shadow_neural=0)
    replay_stroke_world(
        None,
        p.teacher,
        dict(case=dict(source=stroke_foot)),
        103,
        np.zeros((2, 6)),
        dict(kick=kick),
        counts,
        shadow_student=p.student,
        prefix_controls=40,
        student_source=dict(NN_obs=observation[None], NN_action=actual[0].numpy()),
        stroke_foot=stroke_foot,
    )
    assert counts == dict(neural=1, shadow_neural=1)
