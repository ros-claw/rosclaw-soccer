from __future__ import annotations

import inspect

from rosclaw_soccer.training.playmaker_pass_holdout_exam import (
    fresh_playmaker_pass_holdouts,
    run_playmaker_pass_holdout_exam,
    validate_playmaker_pass_holdout_exam,
)


def test_fresh_playmaker_holdouts_are_distinct() -> None:
    contexts = fresh_playmaker_pass_holdouts()
    assert len(contexts) == 3
    assert len({context.context_hash for context in contexts}) == len(contexts)
    assert {context.case_id for context in contexts} == {
        "s132.fresh.00",
        "s132.fresh.01",
        "s132.fresh.02",
    }


def test_playmaker_holdout_api_requires_explicit_frozen_lineage() -> None:
    parameters = inspect.signature(run_playmaker_pass_holdout_exam).parameters
    assert all(value.kind is inspect.Parameter.KEYWORD_ONLY for value in parameters.values())
    assert {
        "actor_path",
        "finisher_actor_path",
        "handoff_actor_path",
        "discovery_report_path",
        "output_dir",
    } <= set(parameters)
    assert callable(validate_playmaker_pass_holdout_exam)
