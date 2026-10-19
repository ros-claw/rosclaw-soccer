"""Whole-six bounded recorded-target replay admission, not causal policy proof."""

import pytest

from scripts.rsi_replay_early_h300_physics import require_scope


def scope():
    return dict(
        profile="six-recorded-early-h300-physics-v1",
        mode="SIM_ONLY",
        source_cases=6,
        physical_step_budget=18000,
        new_neural_inferences=0,
        new_optimizer_updates=0,
        teacher_qualified=False,
        hardware_authorized=False,
        promotion_authorized=False,
        automatic_retry=False,
        independent_neural_or_search_qualification_claimed=False,
    )


def test_whole_six_no_learning_physics_scope():
    require_scope(scope())


@pytest.mark.parametrize("key", list(scope()))
def test_missing_or_changed_scope_is_not_a_different_experiment(key):
    cfg = scope()
    del cfg[key]
    with pytest.raises(ValueError):
        require_scope(cfg)
    cfg = scope()
    value = cfg[key]
    cfg[key] = int(value) if type(value) is bool else float(value) if type(value) is int else "REAL"
    with pytest.raises(ValueError):
        require_scope(cfg)
