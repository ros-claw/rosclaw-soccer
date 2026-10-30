from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path

import pytest

from rosclaw_soccer.growth.locomotion_contact_teacher import G1LocomotionContactTeacherConfig
from rosclaw_soccer.skills.team.independent_team_world import (
    IndependentTeamWorldConfig,
    IndependentTeamWorldScenario,
    _scope_receive_prediction,
    simulate_independent_team_world,
)
from rosclaw_soccer.training.four_vs_four_match import build_four_vs_four_fixture


def test_disabled_scope_preserves_existing_config_identity() -> None:
    assert IndependentTeamWorldConfig().config_hash == (
        "sha256:fd442bce83f737c64e2ee59ecc09dc204376100f9476a21361b41b538d8c41d1"
    )


def test_prediction_scope_limits_only_selected_players() -> None:
    teacher = replace(
        G1LocomotionContactTeacherConfig(),
        receive_velocity_prediction_sec=0.08,
        receive_velocity_prediction_lateral_only=True,
    )
    ids = ("red.playmaker",)
    assert (
        _scope_receive_prediction(teacher, agent_id="red.playmaker", eligible_agent_ids=ids)
        is teacher
    )
    assert (
        _scope_receive_prediction(teacher, agent_id="blue.finisher", eligible_agent_ids=None)
        is teacher
    )
    other = _scope_receive_prediction(teacher, agent_id="blue.finisher", eligible_agent_ids=ids)
    assert other == replace(
        teacher, receive_velocity_prediction_sec=0.0, receive_velocity_prediction_lateral_only=False
    )


@pytest.mark.parametrize(
    "ids", [(), ("red.playmaker", "red.playmaker"), ("red.playmaker", "blue.playmaker"), ("!",)]
)
def test_prediction_scope_rejects_empty_duplicate_or_noncanonical_ids(ids: tuple[str, ...]) -> None:
    with pytest.raises(ValueError, match="canonical agent identities"):
        IndependentTeamWorldConfig(receive_prediction_agent_ids=ids)


def test_prediction_scope_changes_bound_config_identity() -> None:
    base = IndependentTeamWorldConfig()
    scoped = replace(base, receive_prediction_agent_ids=("red.playmaker",))
    assert scoped.config_hash != base.config_hash


@pytest.mark.parametrize(
    ("agent", "teacher"),
    [
        (
            "unknown.player",
            replace(G1LocomotionContactTeacherConfig(), receive_velocity_prediction_sec=0.04),
        ),
        ("red.playmaker", None),
        ("red.playmaker", G1LocomotionContactTeacherConfig()),
    ],
)
def test_world_rejects_scope_outside_live_teacher_roster(
    agent: str, teacher: G1LocomotionContactTeacherConfig | None
) -> None:
    asset = os.environ.get("ROSCLAW_G1_ASSET_ROOT")
    if not asset:
        pytest.skip("ROSCLAW_G1_ASSET_ROOT is not configured")
    fixture = build_four_vs_four_fixture(Path(asset), basic_ball_play=True)
    scenario = IndependentTeamWorldScenario(
        scenario_id="s199.test.scope.reject",
        ball_initial_position_m=(3.0, 0.0, fixture.goal.ball_radius_m),
        ball_initial_velocity_mps=(0.0, 0.0, 0.0),
        seed=1,
    )
    with pytest.raises(ValueError, match="scope requires roster members and an active teacher"):
        simulate_independent_team_world(
            asset_root=Path(asset),
            roster=fixture.roster,
            cells=fixture.cells,
            players=fixture.players,
            scenario=scenario,
            goal=fixture.goal,
            config=IndependentTeamWorldConfig(receive_prediction_agent_ids=(agent,)),
            contact_teacher_config=teacher,
        )
