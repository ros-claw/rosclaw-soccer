import copy

import pytest
from test_joint_delivery_teacher import row
from test_parallel_refinement_batch import fixture

from rosclaw_soccer.rsi.parallel_refinement_batch import (
    FULL_OWN,
    FULL_SOURCE,
    ROLE_OWN,
    ROLE_SOURCE,
    pair_plan,
    parallel_config,
)
from rosclaw_soccer.rsi.receiver_role_review import paired_role_review
from rosclaw_soccer.rsi.refinement_actor_clock import DIAGNOSTIC, FULL


def own(source, role):
    result = copy.deepcopy(source)
    result.update(
        status=ROLE_OWN if role else FULL_OWN,
        source_hash="new-source" if role else "factual-source",
        independent_whole_world_replay=True,
        sampling_law_audit_passed=True,
        records=[{k: v for k, v in r.items() if k != "artifacts"} for r in result["records"]],
    )
    return result


def reports():
    """Synthetic accounting fixture ONLY, never a physical source/own proof."""
    factual, _ = fixture()
    cfg = parallel_config(26, 32, full_recovery=True)
    courses = [factual["records"][i]["course"] | dict(state_id=26) for i in (0, 32, 64)]
    factual.update(
        config=cfg,
        status=FULL_SOURCE,
        source_head="old-code",
        training_hash="same-A",
        optimizer_updates=0,
        Fresh_verified=False,
        team_game_verified=False,
        actor_training_admitted=False,
        hardware_authorized=False,
        promotion_authorized=False,
        goal_complete=False,
    )
    for pair in range(48):
        for plan in pair_plan(cfg, courses, pair):
            r = factual["records"][plan["index"]]
            r.update(
                plan,
                controller_kind=FULL,
                stochastic_controls=389,
                deterministic_recovery_excluded_from_actor_rows=False,
                outcome=row(0.5, 2.7)["outcome"],
                artifacts=dict(initial=dict(hash="initial" + str(plan["index"] // 32))),
            )
    new = copy.deepcopy(factual)
    new.update(
        config=parallel_config(26, 32, full_recovery=True, receiver_role_ablation=True),
        status=ROLE_SOURCE,
        source_head="new-code",
    )
    for r in new["records"]:
        r.update(
            controller_kind=DIAGNOSTIC,
            receiver_role_ablation=True,
            receiver_intent_mode="anchor-plane-v2",
        )
    for i in (6, 24):
        new["records"][i]["outcome"] = row(0.05, 2.7)["outcome"]
    return factual, new


def review(factual, new):
    return paired_role_review(
        factual,
        own(factual, False),
        new,
        own(new, True),
        factual_source_hash="factual-source",
        source_hash="new-source",
    )


def test_known_two_rescues_are_not_96_unseen_successes_or_neural_growth():
    result = review(*reports())
    assert result["overall"]["strict_rescues"] == [6, 24]
    assert result["known_selected_2"]["strict_new"] == 2
    assert result["remaining_94"]["strict_new"] == 0
    assert not result["broader_consumed_TRAIN_role_support"]
    assert result["unique_consumed_TRAIN_initials"] == 3
    assert result["next_agent_action"] == "ONLY_KNOWN_LOCAL_RESCUES_EXPAND_DISTINCT_TRAIN_COURSES"
    assert all(
        result[k] is False
        for k in (
            "Fresh_verified",
            "promotion_authorized",
            "goal_complete",
            "receiver_neural_learning_proven",
            "team_game_verified",
        )
    )


def test_additional_different_initial_rescue_supports_learning_design_not_promotion():
    f, n = reports()
    n["records"][33]["outcome"] = row(0.1, 3.0)["outcome"]
    r = review(f, n)
    assert r["remaining_94"]["strict_rescues"] == [33]
    assert r["broader_consumed_TRAIN_role_support"]
    assert not r["promotion_authorized"] and not r["Fresh_verified"]


@pytest.mark.parametrize("kind", ["safety", "functional", "strict"])
def test_regression_cannot_hide_behind_selected_rescues(kind):
    f, n = reports()
    if kind == "safety":
        n["records"][50]["outcome"]["passer_clean_foot"] = False
    elif kind == "functional":
        n["records"][50]["outcome"] = row(0.5, 2.7, received=False)["outcome"]
    else:
        f["records"][50]["outcome"] = row(0.05, 2.7)["outcome"]
    r = review(f, n)
    assert not r["broader_consumed_TRAIN_role_support"]
    assert r["next_agent_action"] in (
        "REJECT_SAFETY_REGRESSION_REVIEW_CAUSE",
        "REJECT_STRICT_REGRESSION_REVIEW_CAUSE",
        "REPAIR_RECEIVING_RETENTION_BEFORE_LEARNING_OR_PROMOTION",
    )


@pytest.mark.parametrize(
    "fault",
    [
        "missing",
        "seed",
        "initial",
        "course",
        "model",
        "law",
        "own",
        "clock",
        "fresh",
        "label",
        "nan",
    ],
)
def test_partial_unpaired_or_unverified_reports_rejected(fault):
    f, n = reports()
    if fault == "missing":
        n["records"].pop()
    elif fault == "seed":
        n["records"][0]["sampling_seed"] += 1
    elif fault == "initial":
        n["records"][0]["artifacts"]["initial"]["hash"] = "different"
    elif fault == "course":
        n["records"][0]["course"] = n["records"][32]["course"]
    elif fault == "model":
        n["training_hash"] = "changed-A"
    elif fault == "fresh":
        n["Fresh_verified"] = True
    elif fault == "clock":
        n["counts"]["native_steps"] += 1
    elif fault == "label":
        n["records"][6]["outcome"]["combined_precision_and_functional_receive"] = False
    elif fault == "nan":
        n["records"][6]["outcome"]["incoming_arrival_speed_mps"] = float("nan")
    fo, no = own(f, False), own(n, True)
    if fault == "law":
        no["sampling_law_audit_passed"] = False
    if fault == "own":
        no["records"][0]["outcome"]["whole_body_healthy"] = False
    with pytest.raises(ValueError):
        paired_role_review(
            f, fo, n, no, factual_source_hash="factual-source", source_hash="new-source"
        )
