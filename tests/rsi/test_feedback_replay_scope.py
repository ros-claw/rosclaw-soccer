"""Fixed independent replay admission, not candidate performance evidence."""

import pytest

from scripts.rsi_replay_feedback_screen import require_scope


def scope():
    return dict(
        mode="SIM_ONLY",
        profile="six-recorded-feedback-replay-v1",
        source_cases=6,
        recorded_step_budget=18000,
        branch_step_budget=1680,
        combined_step_budget=19680,
        new_neural_inferences=0,
        new_optimizer_updates=0,
        new_closed_loop_policy_trials=0,
        hardware_authorized=False,
        promotion_authorized=False,
        training_authorized=False,
        teacher_qualified=False,
        private_fresh_accessed=False,
    )


def test_exact_scope():
    require_scope(scope())


@pytest.mark.parametrize("key", list(scope()))
def test_no_missing_or_widened_independent_replay_scope(key):
    cfg = scope()
    del cfg[key]
    with pytest.raises(ValueError):
        require_scope(cfg)
    cfg = scope()
    value = cfg[key]
    cfg[key] = int(value) if type(value) is bool else float(value) if type(value) is int else "REAL"
    with pytest.raises(ValueError):
        require_scope(cfg)
