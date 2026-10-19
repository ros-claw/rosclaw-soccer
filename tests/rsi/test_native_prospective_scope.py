"""Full-horizon pilot limits and exact original-prefix checks, not skill gains."""

import numpy as np
import pytest

from scripts.rsi_native_prospective_impact import require_prefix, require_scope


def scope():
    return dict(
        mode="SIM_ONLY",
        profile="twelve-native-prospective-impact-v1",
        source_cases=6,
        new_native_episodes=12,
        native_step_budget=36000,
        forecast_step_budget=5400,
        combined_step_budget=41400,
        new_independent_cpu_episodes=0,
        official_horizon_frames=300,
        post_contact_frames=60,
        new_optimizer_updates=0,
        hardware_authorized=False,
        promotion_authorized=False,
        training_authorized=False,
        teacher_qualified=False,
        private_fresh_accessed=False,
        independent_qualification_claimed=False,
    )


def test_exact_scope():
    require_scope(scope())


@pytest.mark.parametrize("key", list(scope()))
def test_no_widened_budget_or_qualification_claim(key):
    v = scope()
    del v[key]
    with pytest.raises(ValueError):
        require_scope(v)
    v = scope()
    old = v[key]
    v[key] = int(old) if type(old) is bool else float(old) if type(old) is int else "FRESH"
    with pytest.raises(ValueError):
        require_scope(v)


def test_actual_intervention_frame_only_changes_after_step_arrays():
    original = dict(
        canonical_qpos=np.zeros((300, 1, 43)),
        torque_nm=np.zeros((300, 1, 10, 29)),
        foundation_neural_decoder_input=np.zeros((300, 1, 994), dtype=np.float32),
    )
    changed = {k: v.copy() for k, v in original.items()}
    changed["torque_nm"][100:] = 1
    changed["canonical_qpos"][101:] = 1
    changed["foundation_neural_decoder_input"][101:] = 1
    require_prefix(changed, original, 100)
    for key in original:
        bad = {k: v.copy() for k, v in changed.items()}
        bad[key][99] = 1
        with pytest.raises(ValueError):
            require_prefix(bad, original, 100)
    changed["canonical_qpos"][100] = 1
    with pytest.raises(ValueError):
        require_prefix(changed, original, 100)


def test_zero_control_requires_all300_original_bits_and_whole_shapes():
    original = dict(canonical_qpos=np.zeros((300, 1, 43)))
    require_prefix(original, original, None)
    for changed in (
        {},
        dict(canonical_qpos=np.zeros((299, 1, 43))),
        dict(canonical_qpos=np.full((300, 1, 43), -0.0)),
    ):
        with pytest.raises(ValueError):
            require_prefix(changed, original, None)
