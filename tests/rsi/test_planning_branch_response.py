import copy

import numpy as np
import pytest

from rosclaw_soccer.rsi.observed_ball_response import predict_matched_effect
from rosclaw_soccer.rsi.planning_branch_response import (
    fit_planning_response,
    predict_planning_response,
)
from rosclaw_soccer.sim.contracts import hash_json


@pytest.fixture
def fitted():
    baseline = np.zeros((3, 17))
    action = baseline.copy()
    action[:, 12] = 1
    y = np.zeros((3, 3))
    model = fit_planning_response(
        baseline,
        action,
        y,
        y + 0.1,
        source_dataset_hash="sha256:" + "1" * 64,
        seed=1019,
        steps=2,
    )
    return model, baseline, action


def test_planning_domain_separate_and_zero_boundary_exact(fitted):
    model, baseline, action = fitted
    assert model["response_model"]["optimizer_steps"] == 6
    assert model["guarded_execution_response_qualified"] is False
    np.testing.assert_array_equal(predict_planning_response(model, baseline), np.zeros((3, 3)))
    assert predict_planning_response(model, action).shape == (3, 3)
    with pytest.raises(ValueError):
        predict_matched_effect(model, action)
    with pytest.raises(ValueError):
        predict_planning_response(model["response_model"], action)


@pytest.mark.parametrize(
    "field,value",
    [
        ("conditioning_law", "ACTUAL_GUARDED_RELEASE"),
        ("guarded_execution_response_qualified", True),
        ("hardware_authorized", True),
    ],
)
def test_rehashed_domain_or_authority_relabelling_refused(fitted, field, value):
    model, _, action = fitted
    bad = copy.deepcopy(model)
    bad[field] = value
    bad["model_hash"] = hash_json({k: v for k, v in bad.items() if k != "model_hash"})
    with pytest.raises(ValueError):
        predict_planning_response(bad, action)
