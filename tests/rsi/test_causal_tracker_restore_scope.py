"""Bounded source-model diagnostic contract; not physical skill evidence."""

import pytest

from scripts.rsi_qualify_causal_tracker_restore import require_scope


def scope():
    return dict(
        mode="SIM_ONLY",
        profile="six-causal-tracker-restore-v1",
        source_cases=6,
        maximum_planner_run_calls=60,
        current_encoder_decoder_pairs=6,
        next_golden_encoder_decoder_pairs=6,
        maximum_constructor_parity_probe_passes=16,
        new_physical_steps=0,
        new_closed_loop_policy_trials=0,
        new_optimizer_updates=0,
        hardware_authorized=False,
        promotion_authorized=False,
        private_fresh_accessed=False,
        training_authorized=False,
        teacher_qualified=False,
        actor_future_event_input_allowed=False,
    )


def test_fixed_scope():
    require_scope(scope())


@pytest.mark.parametrize("key", list(scope()))
def test_no_missing_or_widened_budget(key):
    cfg = scope()
    del cfg[key]
    with pytest.raises(ValueError):
        require_scope(cfg)
    cfg = scope()
    value = cfg[key]
    cfg[key] = int(value) if type(value) is bool else float(value) if type(value) is int else "REAL"
    with pytest.raises(ValueError):
        require_scope(cfg)
