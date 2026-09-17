import numpy as np
import pytest
from test_s215_near_ball_residual import policy, samples

from rosclaw_soccer.training.near_ball_residual_ppo import update_private_actors


def test_learning_eligibility_does_not_relabel_actions_or_physical_activation():
    p = policy()
    trace = samples(p)
    original = {k: v.copy() for k, v in trace.items()}
    trace["residual_learning_mask"] = np.zeros((40, 8), bool)
    child, rows = update_private_actors(p, [trace])
    assert rows[0]["active_samples"] == 40
    assert rows[0]["learning_samples"] == 0
    assert not any(row["updated"] for row in rows)
    for k in p.weights:
        np.testing.assert_array_equal(child.weights[k], p.weights[k])
    for k in original:
        np.testing.assert_array_equal(trace[k], original[k])


def test_full_eligibility_preserves_numerics_but_binds_dataset_mask():
    p = policy()
    trace = samples(p)
    plain, plain_rows = update_private_actors(p, [trace])
    child, rows = update_private_actors(
        p, [{**trace, "residual_learning_mask": np.ones((40, 8), bool)}]
    )
    for k in p.weights:
        np.testing.assert_array_equal(child.weights[k], plain.weights[k])
    assert rows[0]["learning_samples"] == 40 and rows[0]["updated"]
    assert rows[0]["core_plasticity"] != plain_rows[0]["core_plasticity"]


@pytest.mark.parametrize("mask", [np.zeros((40, 7), bool), np.zeros((40, 8), int)])
def test_learning_mask_requires_explicit_boolean_frame_role_contract(mask):
    p = policy()
    with pytest.raises(ValueError, match="learning"):
        update_private_actors(p, [{**samples(p), "residual_learning_mask": mask}])


def test_mixed_eligibility_contracts_rejected():
    p = policy()
    trace = samples(p)
    with pytest.raises(ValueError, match="learning"):
        update_private_actors(
            p, [trace, {**trace, "residual_learning_mask": np.ones((40, 8), bool)}]
        )
