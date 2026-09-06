from __future__ import annotations

import inspect

from rosclaw_soccer.training.playmaker_pass_repair_discovery import (
    playmaker_repair_actions,
    run_playmaker_pass_repair_discovery,
)


def test_playmaker_repair_search_is_compact_and_distinct() -> None:
    actions = playmaker_repair_actions()
    assert len(actions) == 7
    assert len({action.action_hash for action in actions}) == len(actions)
    assert all(action.stance_correction_x_m == 0.0 for action in actions)
    assert all(action.stance_correction_y_m == 0.0 for action in actions)


def test_playmaker_repair_requires_explicit_failed_lineage() -> None:
    parameters = inspect.signature(run_playmaker_pass_repair_discovery).parameters
    assert all(value.kind is inspect.Parameter.KEYWORD_ONLY for value in parameters.values())
    assert {
        "rejected_holdout_report_path",
        "playmaker_actor_path",
        "finisher_actor_path",
        "handoff_actor_path",
        "output_dir",
    } <= set(parameters)
