import pytest

torch = pytest.importorskip("torch")

from test_kick_prior_ppo import fixture  # noqa: E402

from rosclaw_soccer.rsi.kick_causal_learning import CausalKickLearner, copy_actor  # noqa: E402


def example():
    actor, x, _, _, episode = fixture()
    episode.update(returns=torch.linspace(-1, 1, 5), frames=torch.arange(100, 105))
    learner = CausalKickLearner(actor, x)
    return learner, episode


def test_physical_rollback_restores_actor_and_adam_but_keeps_valid_value_learning():
    learner, episode = example()
    before = copy_actor(learner.actor)
    result = learner.propose([episode])
    assert result["actor_decision"]["accepted"]
    assert learner.pending is not None and learner.value_steps == 4
    with pytest.raises(ValueError, match="physical decision"):
        learner.propose([episode])
    assert len(learner.actor_optimizer.state) == 2
    learner.physical_decision(False)
    assert not learner.actor_optimizer.state
    assert all(torch.equal(v, learner.actor.state_dict()[k]) for k, v in before.items())
    assert learner.pending is None and learner.motor_commits == 0
    assert learner.value_steps == 4 and len(learner.value_optimizer.state) == 4
    with pytest.raises(ValueError):
        learner.physical_decision(True)


def test_commit_preserves_moments_and_does_not_unfreeze_original_features():
    learner, episode = example()
    before = copy_actor(learner.actor)
    assert learner.propose([episode])["actor_decision"]["accepted"]
    learner.physical_decision(True)
    assert learner.motor_commits == 1 and learner.pending is None
    assert [int(s["step"]) for s in learner.actor_optimizer.state.values()] == [1, 1]
    assert any(not torch.equal(v, learner.actor.state_dict()[k]) for k, v in before.items())
    assert all(
        torch.equal(v, learner.actor.state_dict()[k])
        for k, v in before.items()
        if not k.startswith("actor.6.")
    )


@pytest.mark.parametrize("defect", ["teacher", "unexecuted", "stale_policy", "nan_return"])
def test_invalid_batch_rejected_before_value_update(defect):
    learner, episode = example()
    if defect == "teacher":
        episode["external_teacher"] = True
    elif defect == "unexecuted":
        episode["actual_pd_commitment_verified"] = False
    elif defect == "stale_policy":
        episode["old_means"] = episode["old_means"] + 0.1
    else:
        episode["returns"][0] = float("nan")
    with pytest.raises(ValueError):
        learner.propose([episode])
    assert learner.value_steps == 0 and not learner.value_optimizer.state
