from __future__ import annotations

from dataclasses import asdict, replace

import pytest

from rosclaw_soccer.training.contextual_finish_portfolio import (
    ContextualFinishPortfolioConfig,
    FinishPortfolioContext,
    _config_from_dict,
    _precise,
    _stability_retained,
)


def test_default_portfolio_has_disjoint_growth_and_holdout_partitions() -> None:
    config = ContextualFinishPortfolioConfig()
    contexts = (
        *config.success_discovery,
        *config.failure_discovery,
        *config.success_holdouts,
        *config.failure_holdouts,
    )

    assert len(contexts) == 12
    assert len({context.case_id for context in contexts}) == 12
    assert (
        len(
            {
                (context.receiver_phase_start_sec, context.receiver_lateral_lane_m)
                for context in contexts
            }
        )
        == 12
    )
    assert (
        len(config.success_policy_target_y_candidates_m)
        * len(config.success_foot_yaw_candidates_rad)
        == 16
    )
    assert (
        len(config.failure_policy_target_y_candidates_m)
        * len(config.failure_foot_yaw_candidates_rad)
        == 30
    )


def test_duplicate_holdout_coordinate_is_rejected() -> None:
    config = ContextualFinishPortfolioConfig()
    duplicate = FinishPortfolioContext("duplicate-sealed", 1.900, 0.079)

    with pytest.raises(ValueError, match="portfolio config is invalid"):
        replace(config, failure_holdouts=(duplicate, config.failure_holdouts[1]))


def test_failure_search_budget_cannot_expand_silently() -> None:
    with pytest.raises(ValueError, match="portfolio config is invalid"):
        ContextualFinishPortfolioConfig(
            failure_foot_yaw_candidates_rad=(0.03, 0.04, 0.05, 0.06, 0.07, 0.08)
        )


def test_config_json_round_trip_preserves_hash() -> None:
    config = ContextualFinishPortfolioConfig()

    restored = _config_from_dict(asdict(config))

    assert restored == config
    assert restored.config_hash == config.config_hash


def test_precision_gate_never_accepts_unsafe_low_error() -> None:
    config = ContextualFinishPortfolioConfig()
    row = {"safe": False, "result": {"goal_crossed": True, "target_error_m": 0.01}}

    assert not _precise(row, config)
    assert _precise(
        {"safe": True, "result": {"goal_crossed": True, "target_error_m": 0.10}},
        config,
    )


def test_stability_gate_checks_pelvis_and_support_slip() -> None:
    config = ContextualFinishPortfolioConfig()
    parent = {
        "shooter_min_pelvis_height_m": 0.65,
        "shooter_post_contact_support_foot_slip_m": 0.08,
    }

    assert _stability_retained(
        {
            "shooter_min_pelvis_height_m": 0.64,
            "shooter_post_contact_support_foot_slip_m": 0.12,
        },
        parent,
        config,
    )
    assert not _stability_retained(
        {
            "shooter_min_pelvis_height_m": 0.60,
            "shooter_post_contact_support_foot_slip_m": 0.12,
        },
        parent,
        config,
    )
    assert not _stability_retained(
        {
            "shooter_min_pelvis_height_m": 0.65,
            "shooter_post_contact_support_foot_slip_m": 0.17,
        },
        {
            "shooter_min_pelvis_height_m": 0.65,
            "shooter_post_contact_support_foot_slip_m": 0.15,
        },
        config,
    )
