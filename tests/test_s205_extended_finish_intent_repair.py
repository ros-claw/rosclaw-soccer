from __future__ import annotations

from dataclasses import asdict, replace

import pytest

from rosclaw_soccer.training.extended_finish_intent_repair import (
    ExtendedFinishIntentRepairConfig,
    _config_from_dict,
    _refinement_seed_key,
    _selection_key,
)


def test_default_repair_plan_is_deterministic_content_bound_and_sim_only() -> None:
    config = ExtendedFinishIntentRepairConfig()

    first = config.plan.local_candidates(config.warm_start)
    second = config.plan.local_candidates(config.warm_start)

    assert first == second
    assert len(first) == 33
    assert first[0].stage == "WARM_START"
    assert first[0].values == config.warm_start
    assert len({candidate.candidate_hash for candidate in first}) == 33
    assert all(candidate.activation_ceiling == "SIM_ONLY" for candidate in first)
    assert all(not candidate.hardware_authorized for candidate in first)
    assert all(not candidate.direct_joint_torque_output for candidate in first)
    assert config.refinement_plan.plan_hash != config.plan.plan_hash
    assert len(config.refinement_plan.local_candidates(config.warm_start)) == 33


def test_repair_config_round_trip_preserves_hash() -> None:
    config = ExtendedFinishIntentRepairConfig()

    restored = _config_from_dict(asdict(config))

    assert restored == config
    assert restored.config_hash == config.config_hash


def test_repair_budget_and_hardware_authority_are_fail_closed() -> None:
    config = ExtendedFinishIntentRepairConfig()

    with pytest.raises(ValueError, match="repair config is invalid"):
        replace(config, local_candidate_count=64)
    with pytest.raises(ValueError, match="repair config is invalid"):
        replace(config, hardware_authorized=True)
    with pytest.raises(ValueError, match="repair config is invalid"):
        replace(config, warm_start=(0.50, 0.05, -0.04, 0.03))


def test_selection_prefers_safe_goal_crossing_before_low_untrusted_error() -> None:
    unsafe = {
        "safe": False,
        "candidate": {"candidate_index": 0},
        "result": {
            "goal_crossed": True,
            "target_error_m": 0.001,
            "pass_delivery_error_m": 0.001,
            "shooter_post_contact_support_foot_slip_m": 0.01,
            "shooter_min_pelvis_height_m": 0.70,
        },
    }
    safe = {
        "safe": True,
        "candidate": {"candidate_index": 1},
        "result": {
            "goal_crossed": True,
            "target_error_m": 0.05,
            "pass_delivery_error_m": 0.02,
            "shooter_post_contact_support_foot_slip_m": 0.10,
            "shooter_min_pelvis_height_m": 0.65,
        },
    }

    assert min((unsafe, safe), key=_selection_key) is safe


def test_refinement_seed_prefers_stable_goal_over_unstable_low_error() -> None:
    unstable = {
        "safe": True,
        "stability_retained": False,
        "candidate": {"candidate_index": 0},
        "result": {"goal_crossed": True, "target_error_m": 0.01},
    }
    stable = {
        "safe": True,
        "stability_retained": True,
        "candidate": {"candidate_index": 1},
        "result": {"goal_crossed": True, "target_error_m": 0.11},
    }

    assert min((unstable, stable), key=_refinement_seed_key) is stable
