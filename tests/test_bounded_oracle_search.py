import numpy as np
import pytest

from rosclaw_soccer.training.bounded_oracle_search import (
    OracleEvaluation,
    refine_oracle_population,
)


def test_policy_incumbent_is_not_a_zero_vector_elite():
    proposals = (
        OracleEvaluation((0.8, 0.8), False, True, 2.0),
        OracleEvaluation((0.9, 0.9), False, True, 1.0),
    )
    incumbent = OracleEvaluation(None, True, True, 100.0)
    assert refine_oracle_population(proposals, seed=1) == refine_oracle_population(
        (incumbent, *proposals), seed=1
    )
    assert max((incumbent, *proposals), key=lambda row: row.rank) is incumbent


def test_normalized_determinism_and_safety_first():
    unsafe = OracleEvaluation((-1.0,), True, False, 1000.0)
    safe = (OracleEvaluation((1.0,), False, True, 0.0), OracleEvaluation((1.0,), False, True, -1.0))
    output = refine_oracle_population((unsafe, *safe), seed=12)
    assert np.asarray(output).shape == (8, 1)
    assert np.min(output) > 0.5 and np.max(output) <= 1
    assert output == refine_oracle_population((unsafe, *safe), seed=12)


@pytest.mark.parametrize("proposal", [(float("nan"),), (1.01,), (), [0.0]])
def test_invalid_proposals(proposal):
    with pytest.raises(ValueError):
        OracleEvaluation(proposal, False, True, 0.0)


def test_dimensions_and_nonvector_only_rejected():
    with pytest.raises(ValueError):
        refine_oracle_population(
            (
                OracleEvaluation((0.0,), False, True, 0.0),
                OracleEvaluation((0.0, 0.0), False, True, 0.0),
            ),
            seed=1,
        )
    with pytest.raises(ValueError):
        refine_oracle_population((OracleEvaluation(None, False, True, 0.0),) * 2, seed=1)
