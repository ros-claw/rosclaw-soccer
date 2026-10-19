"""Whole causal calibration contract/comparison; not actual G1 task evidence."""

import numpy as np
import pytest

from scripts.rsi_calibrate_causal_impact_replay import compare_frame, compare_outcome, require_scope


def scope():
    return dict(
        mode="SIM_ONLY",
        profile="six-full-causal-replay-zero-impact-v1",
        impact_enabled=False,
        source_cases=6,
        full_causal_rollouts=6,
        native_step_budget=18000,
        forecast_step_budget=0,
        official_horizon_frames=300,
        post_contact_frames=60,
        new_policy_candidates=0,
        new_optimizer_updates=0,
        hardware_authorized=False,
        promotion_authorized=False,
        teacher_qualified=False,
        training_authorized=False,
        private_fresh_accessed=False,
        actor_future_event_input_allowed=False,
    )


def test_exact_scope():
    require_scope(scope())


@pytest.mark.parametrize("key", list(scope()))
def test_cannot_drop_budget_or_enable_interventions_or_hardware(key):
    cfg = scope()
    del cfg[key]
    with pytest.raises(ValueError):
        require_scope(cfg)
    cfg = scope()
    old = cfg[key]
    cfg[key] = int(old) if type(old) is bool else float(old) if type(old) is int else "FRESH"
    with pytest.raises(ValueError):
        require_scope(cfg)


def test_neural_capture_requires_all_original_bits_even_signed_zero():
    original = dict(foundation_neural_decoder_input=np.zeros((300, 1, 994), dtype=np.float32))
    actual = {k: a[0].copy() for k, a in original.items()}
    compare_frame(actual, original, 0)
    actual["foundation_neural_decoder_input"][0, 0] = -0.0
    with pytest.raises(ValueError, match="bits"):
        compare_frame(actual, original, 0)


def test_all_fields_shapes_types_and_full_numeric_physics_checked():
    original = dict(canonical_qpos=np.zeros((300, 1, 43)))
    for a in (
        {},
        dict(canonical_qpos=np.zeros((1, 42))),
        dict(canonical_qpos=np.zeros((1, 43), dtype=np.float32)),
        dict(canonical_qpos=np.ones((1, 43))),
        dict(canonical_qpos=np.full((1, 43), np.nan)),
    ):
        with pytest.raises(ValueError):
            compare_frame(a, original, 0)


def test_outcomes_keep_labels_and_nulls_exact_with_original_physics_tolerance():
    original = dict(
        task_pass=False, first_foot_contact_frame=187, forward_60_m=2.3, lateral_60_m=None
    )
    compare_outcome(dict(original), original)
    for k, v in (
        ("task_pass", True),
        ("task_pass", 0),
        ("first_foot_contact_frame", 188),
        ("forward_60_m", 2.31),
        ("forward_60_m", float("nan")),
        ("forward_60_m", float("inf")),
        ("lateral_60_m", 0.0),
    ):
        with pytest.raises(ValueError):
            compare_outcome(dict(original, **{k: v}), original)
