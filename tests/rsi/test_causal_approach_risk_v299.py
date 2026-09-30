"""Risk learner uses only the frozen, explicitly limited model families."""

from scripts.rsi_train_causal_approach_risk_v299 import _models


def test_frozen_risk_model_families() -> None:
    models = _models()
    assert set(models) == {"tree_depth3", "forest_depth4", "logistic_c01"}
    assert models["tree_depth3"].max_depth == 3
    assert models["forest_depth4"].n_estimators == 100
