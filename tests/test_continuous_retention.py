import numpy as np
import pytest
from test_s215_near_ball_residual import policy, samples

from rosclaw_soccer.training.continuous_retention import first_batch_anchor
from rosclaw_soccer.training.near_ball_residual_ppo import update_private_actors


def test_first_training_batch_anchor_is_immutable_reproducible_and_supports_geometry():
    reference = policy().with_task_geometry()
    trace = samples(policy())
    trace["residual_observations"] = np.pad(
        trace["residual_observations"], ((0, 0), (0, 0), (0, 2))
    )
    anchor = first_batch_anchor(reference, [trace], coefficient=100.0)
    assert anchor.policy.policy_hash == reference.policy_hash
    replay = first_batch_anchor(reference, [trace], coefficient=100.0)
    assert anchor.anchor_hash == replay.anchor_hash
    original = anchor.observations.copy()
    trace["residual_observations"][:] = 0
    np.testing.assert_array_equal(anchor.observations, original)
    assert first_batch_anchor(reference, [], coefficient=0.0) is None
    with pytest.raises(ValueError):
        first_batch_anchor(reference, [], coefficient=100.0)


@pytest.mark.parametrize("bad", [True, -1.0, 0.001, 1001.0, float("nan")])
def test_bad_retention_strength_fails_before_loading_training_data(bad):
    with pytest.raises(ValueError, match="coefficient"):
        first_batch_anchor(policy(), [], coefficient=bad)


def test_anchor_cannot_silently_mix_observation_versions():
    pytest.importorskip("torch")
    p = policy()
    trace = samples(p)
    anchor = first_batch_anchor(p, [trace], coefficient=10.0)
    with pytest.raises(ValueError, match="anchor"):
        update_private_actors(p.with_task_geometry(), [trace], behavior_anchor=anchor)
