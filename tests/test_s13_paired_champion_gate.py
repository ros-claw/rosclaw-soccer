from __future__ import annotations

from dataclasses import replace

import pytest

from rosclaw_soccer.growth.paired_champion_gate import (
    ChampionMetricSpec,
    ChampionSnapshot,
    decision_payload,
    evaluate_paired_champion,
)

_PARENT = "sha256:" + "1" * 64
_CHILD = "sha256:" + "2" * 64
_SUITE = "sha256:" + "3" * 64


def _snapshot(*, child: bool, save: float, angular: float) -> ChampionSnapshot:
    return ChampionSnapshot(
        artifact_hash=_CHILD if child else _PARENT,
        parent_artifact_hash=_PARENT if child else None,
        scenario_suite_hash=_SUITE,
        episode_count=96,
        metrics=(("save_rate", save), ("angular_speed", angular)),
        qualified=True,
    )


def _specs() -> tuple[ChampionMetricSpec, ...]:
    return (
        ChampionMetricSpec("save_rate", "MAXIMIZE", hard_lower_bound=0.0),
        ChampionMetricSpec("angular_speed", "MINIMIZE", hard_upper_bound=3.5),
    )


def test_paired_champion_replaces_only_a_dominating_parent_bound_child() -> None:
    parent = _snapshot(child=False, save=0.30, angular=3.0)
    child = _snapshot(child=True, save=0.35, angular=2.8)
    decision = evaluate_paired_champion(parent=parent, candidate=child, metrics=_specs())
    assert decision.replace_champion
    assert decision.status == "REPLACE_CHAMPION"
    assert decision_payload(decision, parent=parent, candidate=child, metrics=_specs())[
        "report_hash"
    ].startswith("sha256:")


def test_safe_improvement_branch_is_archived_when_an_objective_regresses() -> None:
    parent = _snapshot(child=False, save=0.30, angular=3.0)
    child = _snapshot(child=True, save=0.36, angular=3.1)
    decision = evaluate_paired_champion(parent=parent, candidate=child, metrics=_specs())
    assert not decision.replace_champion
    assert decision.status == "RETAIN_PARENT_ARCHIVE_CANDIDATE"
    assert "angular_speed_regression_exceeds_budget" in decision.reasons


def test_paired_champion_fails_closed_on_wrong_lineage_or_hardware_claim() -> None:
    parent = _snapshot(child=False, save=0.30, angular=3.0)
    child = _snapshot(child=True, save=0.35, angular=2.8)
    with pytest.raises(ValueError, match="bound"):
        evaluate_paired_champion(
            parent=parent,
            candidate=replace(child, parent_artifact_hash="sha256:" + "4" * 64),
            metrics=_specs(),
        )
    with pytest.raises(ValueError, match="SIM_ONLY"):
        replace(child, hardware_command_sent=True)
