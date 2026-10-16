"""Reject bad planning declarations before model allocation or artifact writes."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from rosclaw_soccer.rsi.first_touch_campaign_worker import qualified_episode
from rosclaw_soccer.rsi.public_first_touch_training_plan import PublicFirstTouchTrainingPlan
from rosclaw_soccer.rsi.public_physics_option_planning import planning_contexts
from rosclaw_soccer.rsi.recurrent_sampling_episode_factory import RecurrentSamplingEpisodeFactory


@pytest.mark.parametrize(
    "case",
    [
        "role_without_context",
        "foreign_context",
        "wrong_group",
        "real_stage",
        "foreign_plan",
        "foreign_factory",
        "unallocated_factory",
    ],
)
def test_bad_planning_admission_has_no_physical_or_artifact_side_effects(tmp_path, case):
    mean = "sha256:" + "1" * 64
    plan = PublicFirstTouchTrainingPlan(tuple(range(20261060, 20261070)), mean, 18)
    context = planning_contexts(plan)[0]
    group = context.source_group
    stage = "PLANNING_SIMULATION"
    factory = object()
    if case == "role_without_context":
        context = None
        stage = "SELECTED_OPTION_EVALUATION"
    elif case == "foreign_context":
        context = object()
    elif case == "wrong_group":
        group += 1
    elif case == "real_stage":
        stage = "REAL"
    elif case == "foreign_plan":
        plan = SimpleNamespace(draws=lambda: (_ for _ in ()).throw(AssertionError("must not call")))
    elif case == "unallocated_factory":
        factory = object.__new__(RecurrentSamplingEpisodeFactory)
    folder = tmp_path / "must-not-exist"
    with pytest.raises(ValueError):
        qualified_episode(
            folder,
            group,
            None,
            plan=plan,
            manifest={},
            assets=SimpleNamespace(MEAN=mean),
            source=Path("absent"),
            core=Path("absent"),
            factory=factory,
            compiler=None,
            auditor=None,
            planning_context=context,
            planning_stage=stage,
        )
    assert not folder.exists()
    assert not list(tmp_path.iterdir())
