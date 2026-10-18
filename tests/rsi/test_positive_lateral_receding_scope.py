import pytest

from rosclaw_soccer.rsi import positive_lateral_receding_scope as scope


def test_fixed_domain_yields_eight_courses_from_four_different_source_seeds():
    result = scope.build_scope("sha256:" + "a" * 64)
    assert len(result["contexts"]) == 8
    assert len(result["proposed_jobs"]) == 16
    assert result["proposed_combined_body_substeps"] == 96000
    assert result["cluster_accounting"]["source_context_clusters"] == 4
    assert len(set(result["source_seed_search"]["selected_seeds"])) == 4
    assert all(20261070 <= r["source_seed"] < 20261070 + 2048 for r in result["contexts"])
    assert not result["execution_authorized"] and not result["neural_training_authorized"]
    assert not result["fresh_exam"] and not result["teacher_qualified"]
    for i, context in enumerate(result["contexts"]):
        position = (0, 1, 4, 5)[i // 2]
        assert context["source_lane"] == 9 + 2 * (position % 4)
        assert context["source_group"] == position * 16 + 12 + (i % 2)


def test_selection_is_reproducible_and_behavior_hash_does_not_select_favorable_physical_metadata():
    first = scope.build_scope("sha256:" + "a" * 64)
    assert first == scope.build_scope("sha256:" + "a" * 64)
    second = scope.build_scope("sha256:" + "b" * 64)
    assert first["source_seed_search"] == second["source_seed_search"]
    assert first["contexts"] == second["contexts"]
    assert first["scope_hash"] != second["scope_hash"]


def test_all_rejected_metadata_retained_and_domain_not_clipped():
    result = scope.build_scope("sha256:" + "a" * 64)
    examined = result["source_seed_search"]["examined_metadata"]
    assert len(examined) <= 2048
    assert [r["seed"] for r in examined] == list(range(20261070, 20261070 + len(examined)))
    assert sum(r["metadata_eligible"] for r in examined) == 4
    assert sum(r["source_registry_filler_not_a_declared_physical_trial"] for r in examined) == 2
    assert len(result["source_plan"]["seeds"]) == 6
    assert len(examined) > 4


def test_exhausted_metadata_budget_rejects_whole_curriculum_without_domain_relaxation(monkeypatch):
    monkeypatch.setattr(scope, "sample_training_courses", lambda seed: ((2.8, 0.0, 0.4),) * 16)
    with pytest.raises(ValueError, match="no implicit domain expansion"):
        scope.build_scope("sha256:" + "a" * 64)
