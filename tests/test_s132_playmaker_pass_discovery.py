from __future__ import annotations

import inspect

import pytest

from rosclaw_soccer.training.playmaker_pass_discovery import (
    PlaymakerPassProbeAction,
    default_playmaker_pass_actions,
    run_playmaker_pass_discovery,
    validate_playmaker_pass_discovery,
)


def test_playmaker_action_search_is_distinct_and_bounded() -> None:
    actions = default_playmaker_pass_actions()
    assert len(actions) == 15
    assert len({action.action_hash for action in actions}) == len(actions)
    assert actions[0] == PlaymakerPassProbeAction()
    assert all(action.activation_ceiling == "SIM_ONLY" for action in actions)


def test_playmaker_action_rejects_unsafe_envelope() -> None:
    with pytest.raises(ValueError, match="SIM-only envelope"):
        PlaymakerPassProbeAction(body_yaw_correction_rad=0.061)
    with pytest.raises(ValueError, match="SIM-only envelope"):
        PlaymakerPassProbeAction(swing_speed_scale=0.79)


def test_playmaker_discovery_api_requires_explicit_lineage_and_output() -> None:
    parameters = inspect.signature(run_playmaker_pass_discovery).parameters
    assert all(value.kind is inspect.Parameter.KEYWORD_ONLY for value in parameters.values())
    assert {
        "asset_root",
        "source_s95_dir",
        "rejected_holdout_report_path",
        "teacher_discovery_report_path",
        "actor_path",
        "handoff_actor_path",
        "output_dir",
    } <= set(parameters)
    assert callable(validate_playmaker_pass_discovery)


def test_playmaker_probe_freezes_finisher_arrival_option() -> None:
    import rosclaw_soccer.training.playmaker_pass_discovery as module

    source = inspect.getsource(module._run_probe)
    assert "shooter_causal_strike_option_config" in source
    assert "maximum_arrival_advance_frames=finisher_action.maximum_arrival_advance_frames" in source
