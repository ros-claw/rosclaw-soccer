from dataclasses import replace

import pytest

from rosclaw_soccer.rsi.public_course_cluster_accounting import cluster_contract
from rosclaw_soccer.rsi.public_first_touch_training_plan import PublicFirstTouchTrainingPlan
from rosclaw_soccer.rsi.public_physics_option_planning import planning_contexts
from rosclaw_soccer.rsi.receding_teacher_scope import receding_contexts


def plan():
    return PublicFirstTouchTrainingPlan(tuple(range(20261060, 20261070)), "sha256:" + "a" * 64, 18)


def test_six_current_courses_are_three_base_scene_clusters_not_six_independent_scenes():
    p = plan()
    result = cluster_contract(p, receding_contexts(p))
    assert result["distinct_derived_courses"] == 6
    assert result["source_context_clusters"] == 3
    assert [r["source_groups"] for r in result["clusters"]] == [[76, 77], [108, 109], [140, 141]]
    assert result["same_cluster_must_share_training_validation_split"]
    assert not result["statistical_independence_proven"]


def test_twenty_two_endpoint_union_has_ten_derived_courses_but_five_base_scene_clusters():
    p = plan()
    contexts = {c.context_hash: c for c in (*planning_contexts(p)[2:], *receding_contexts(p))}
    result = cluster_contract(p, tuple(contexts.values()))
    assert result["distinct_derived_courses"] == 10
    assert result["source_context_clusters"] == 5


@pytest.mark.parametrize("tamper", ["duplicate", "source", "empty", "dict"])
def test_reject_replays_and_changed_source_cluster_identity(tamper):
    p = plan()
    contexts = receding_contexts(p)
    if tamper == "duplicate":
        contexts += (contexts[0],)
    elif tamper == "source":
        contexts = (replace(contexts[0], source_context_id=20), *contexts[1:])
    elif tamper == "empty":
        contexts = ()
    else:
        contexts = (vars(p) if hasattr(p, "__dict__") else {},)
    with pytest.raises(ValueError):
        cluster_contract(p, contexts)
