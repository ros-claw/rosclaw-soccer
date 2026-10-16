import json
from dataclasses import replace

import pytest

from rosclaw_soccer.rsi.initial_context_control_portfolio import (
    ControlCase,
    ControlOutcome,
    InitialContextControlPortfolio,
)
from rosclaw_soccer.sim.contracts import hash_json


def fitted():
    cases = tuple(
        ControlCase(
            (2.2 + i * 0.1, float(i % 2) * 0.1, -0.5),
            (ControlOutcome(True, i % 2 == 0, 1.0), ControlOutcome(True, i % 2 == 1, 2.0)),
            tuple(hash_json([i, j]) for j in range(2)),
        )
        for i in range(4)
    )
    return InitialContextControlPortfolio.fit(
        cases=cases,
        option_hashes=(hash_json("A"), hash_json("B")),
        execution_binding_hash=hash_json("fixed body/world/parent/initial robot state/task"),
    )


def test_fits_different_safe_options_from_success_and_failure_without_query_outcomes():
    model = fitted()
    assert model.selected_options == (0, 1, 0, 1)
    for i, case in enumerate(model.cases):
        query = model.query(case.context, execution_binding_hash=model.execution_binding_hash)
        assert query["option_index"] == i % 2
        assert query["training_context_match"]
        assert not query["teacher_qualified"]
        assert not query["unseen_generalization_proven"]
        assert not query["calibrated_probability_model"]
    assert (
        model.query((2.31, 0.1, -0.5), execution_binding_hash=model.execution_binding_hash)[
            "option_index"
        ]
        == 1
    )
    with pytest.raises(TypeError):
        model.query(
            model.cases[0].context,
            execution_binding_hash=model.execution_binding_hash,
            future_outcome=True,
        )


def test_unsafe_options_never_chosen_even_with_high_reward():
    model = fitted()
    bad = replace(
        model.cases[0],
        outcomes=(ControlOutcome(False, False, 90.0), ControlOutcome(True, False, -3.0)),
    )
    model = replace(model, cases=(bad, *model.cases[1:]))
    assert model.selected_options[0] == 1
    none = replace(bad, outcomes=(ControlOutcome(False, False, -10.0),) * 2)
    model = replace(model, cases=(none, *model.cases[1:]))
    assert (
        model.query(none.context, execution_binding_hash=model.execution_binding_hash)["status"]
        == "UNKNOWN"
    )


def test_unknown_is_not_zero_probability_or_default_execution():
    model = fitted()
    for query in (
        model.query((5.0, 0.0, 0.0), execution_binding_hash=model.execution_binding_hash),
        model.query(
            model.cases[0].context, execution_binding_hash=hash_json("different parent or body")
        ),
    ):
        assert query["status"] == "UNKNOWN"
        assert query["option_index"] is None and query["option_hash"] is None


def test_supported_unfitted_context_can_choose_failed_neighbour_without_authority():
    """Public76 exposed this limitation; support must not be read as competence."""
    model = fitted()
    failed = replace(
        model.cases[0],
        outcomes=(ControlOutcome(False, False, -10.0), ControlOutcome(True, False, -1.0)),
    )
    model = replace(model, cases=(failed, *model.cases[1:]))
    query = model.query(
        (failed.context[0] + 0.001, failed.context[1], failed.context[2]),
        execution_binding_hash=model.execution_binding_hash,
    )
    assert query["status"] == "SIM_RESEARCH_OPTION"
    assert query["reason"] == "unqualified_neighbour_choice_not_success_probability"
    assert query["option_index"] == 1
    assert query["source_training_evidence_hash"] == failed.evidence_hashes[1]
    assert not query["training_context_match"]
    assert failed.outcomes[1].safe and not failed.outcomes[1].task_pass
    for key in (
        "calibrated_probability_model",
        "unseen_generalization_proven",
        "teacher_qualified",
        "promotion_authorized",
        "hardware_authorized",
    ):
        assert query[key] is False
    # Query support says nothing about the unexecuted context's physical safety.
    assert "safe" not in query and "task_pass" not in query


@pytest.mark.parametrize(
    "change", ["partial", "duplicate_context", "replayed_evidence", "fresh", "duplicate_option"]
)
def test_no_partial_matrix_duplicate_training_or_private_fresh_fit(change):
    model = fitted()
    with pytest.raises(ValueError):
        if change == "partial":
            case = replace(
                model.cases[0],
                outcomes=model.cases[0].outcomes[:1],
                evidence_hashes=model.cases[0].evidence_hashes[:1],
            )
            replace(model, cases=(case, *model.cases[1:]))
        elif change == "duplicate_context":
            replace(model, cases=(model.cases[0], model.cases[0], *model.cases[2:]))
        elif change == "replayed_evidence":
            case = replace(model.cases[1], evidence_hashes=model.cases[0].evidence_hashes)
            replace(model, cases=(model.cases[0], case, *model.cases[2:]))
        elif change == "fresh":
            replace(model, partition="sealed_fresh")
        else:
            replace(model, option_hashes=(model.option_hashes[0],) * 2)


def test_json_reload_preserves_fitted_choices_and_contract_not_authority():
    model = fitted()
    raw = json.loads(json.dumps(model.to_dict()))
    reloaded = InitialContextControlPortfolio.from_dict(raw)
    assert reloaded.model_hash == model.model_hash
    assert reloaded.selected_options == model.selected_options
    raw["teacher_qualified"] = True
    with pytest.raises(ValueError):
        InitialContextControlPortfolio.from_dict(raw)


@pytest.mark.parametrize(
    "field,value", [("context", [2.0] * 4), ("outcomes", [{}] * 100), ("evidence_hashes", [])]
)
def test_json_inner_shapes_bounded_before_constructing_fit(field, value):
    raw = json.loads(json.dumps(fitted().to_dict()))
    raw["cases"][0][field] = value
    with pytest.raises(ValueError, match="before fit allocation"):
        InitialContextControlPortfolio.from_dict(raw)


@pytest.mark.parametrize(
    "context",
    [(True, 0.0, 0.0), (2.0, float("nan"), 0.0), (2.0, 0.0, float("inf")), (-1.0, 0.0, 0.0)],
)
def test_invalid_query_contexts_rejected(context):
    model = fitted()
    with pytest.raises(ValueError):
        model.query(context, execution_binding_hash=model.execution_binding_hash)


def test_query_result_or_export_mutation_cannot_change_model():
    model = fitted()
    original = model.model_hash
    exported = model.to_dict()
    exported["cases"][0]["outcomes"][0]["safe"] = False
    query = model.query(model.cases[0].context, execution_binding_hash=model.execution_binding_hash)
    query["option_index"] = 1
    assert model.model_hash == original
