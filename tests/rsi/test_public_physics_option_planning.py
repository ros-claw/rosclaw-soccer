from dataclasses import replace

import pytest

from rosclaw_soccer.rsi.initial_context_control_portfolio import ControlOutcome
from rosclaw_soccer.rsi.public_first_touch_training_plan import PublicFirstTouchTrainingPlan
from rosclaw_soccer.rsi.public_physics_option_planning import (
    PlanningTrial,
    choose_planned_option,
    planning_contexts,
    validate_context_source,
)


def digest(index):
    return "sha256:" + f"{index:064x}"


def plan():
    return PublicFirstTouchTrainingPlan(tuple(range(20261060, 20261070)), digest(1), 18)


def trials():
    return tuple(
        PlanningTrial(
            digest(1),
            i,
            digest(10 + i),
            digest(20 + i),
            digest(30 + i),
            digest(2),
            ControlOutcome(True, passed, float(reward)),
        )
        for i, (passed, reward) in enumerate(((False, 5), (True, 3), (True, 3)))
    )


def choose(values):
    return choose_planned_option(
        values,
        context_hash=digest(1),
        binding_hash=digest(2),
        option_hashes=tuple(digest(10 + i) for i in range(3)),
    )


def test_predeclared_eight_contexts_have_no_outcome_access():
    source = plan()
    contexts = planning_contexts(source)
    assert len(contexts) == 8
    assert len({context.context_hash for context in contexts}) == 8
    assert len({context.sampling_seed for context in contexts}) == 8
    assert [context.source_group for context in contexts] == [64, 65, 68, 69, 72, 73, 76, 77]
    assert contexts == planning_contexts(source)
    for context in contexts:
        validate_context_source(context, source, group=context.source_group)
        original = source.draws()[context.source_group]
        assert context.source_context_id == original.context_id
        assert context.course != original.course
        assert context.sampling_seed == original.sampling_seed


def test_invalid_boundary_perturbation_rejects_entire_declaration_without_clipping():
    source = plan()
    original = source.contract()
    with pytest.raises(ValueError, match="leaves"):
        planning_contexts(source, base_groups=(48, 52, 56, 60))
    assert source.contract() == original


@pytest.mark.parametrize(
    "groups",
    [(64, 64, 72, 76), (True, 68, 72, 76), (65, 68, 72, 76), (64, 68, 72), (156, 160, 164, 168)],
)
def test_bad_or_incomplete_source_groups_rejected(groups):
    with pytest.raises(ValueError):
        planning_contexts(plan(), base_groups=groups)


def test_selection_requires_actual_task_pass_not_partial_reward_and_breaks_ties_stably():
    result = choose(trials())
    assert result["option_index"] == 1
    assert result["planning_trial_count"] == 3
    assert not result["physical_receipts_authenticated_here"]
    assert not result["teacher_qualified"]
    assert not result["training_authorized"]


def test_no_feasible_option_is_a_normal_preserved_failure():
    values = tuple(
        replace(trial, outcome=ControlOutcome(False, False, -10.0)) for trial in trials()
    )
    result = choose(values)
    assert result["status"] == "PLANNING_NO_FEASIBLE_OPTION"
    assert result["option_index"] is None
    assert len(result["planning_trials"]) == 3


@pytest.mark.parametrize(
    "change",
    [
        dict(context_hash=digest(99)),
        dict(option_hash=digest(99)),
        dict(option_index=0),
        dict(binding_hash=digest(99)),
        dict(attempt_hash=digest(20)),
    ],
)
def test_mixed_context_catalog_or_reused_attempt_rejected(change):
    values = list(trials())
    values[1] = replace(values[1], **change)
    with pytest.raises(ValueError):
        choose(tuple(values))


def test_missing_planning_failures_and_evaluation_receipts_rejected():
    with pytest.raises(ValueError):
        choose(trials()[:2])
    with pytest.raises(ValueError, match="evaluation"):
        replace(trials()[0], role="EVALUATION")


def test_same_receipt_bytes_do_not_define_execution_or_independent_context_counts():
    # Candidate runs can have identical numerical outputs. The executor still
    # has to prove distinct attempts; these hashes are synthetic, not receipts.
    values = tuple(replace(trial, receipt_hash=digest(50)) for trial in trials())
    assert choose(values)["planning_trial_count"] == 3


@pytest.mark.parametrize(
    "change", [dict(sampling_seed=1), dict(source_context_id=1), dict(perturbation_index=1)]
)
def test_context_source_metadata_cannot_be_forged(change):
    source = plan()
    context = replace(planning_contexts(source)[0], **change)
    with pytest.raises(ValueError):
        validate_context_source(context, source, group=context.source_group)


def test_context_course_is_exactly_derived_not_an_arbitrary_in_domain_override():
    source = plan()
    original = planning_contexts(source)[0]
    x, y, vx = original.course
    changed = replace(original, course=(x + 0.001, y, vx))
    with pytest.raises(ValueError, match="frozen perturbation"):
        validate_context_source(changed, source, group=original.source_group)
