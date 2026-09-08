"""Symmetric, role-complete 4v4 fixture; both sides share the same controllers."""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

from rosclaw_soccer.growth.independent_agent_cell import build_independent_agent_cell
from rosclaw_soccer.growth.role_self_model import MatchRole, TeamRoleRoster
from rosclaw_soccer.training.independent_team_growth import (
    IndependentTeamFixture,
    build_independent_three_vs_three_fixture,
)
from rosclaw_soccer.world.multi_player import G1PitchPlayerSpec

# Half-turn symmetry about (3, 0), between goal planes -1.5 and 7.5.
FORMATION = (
    (MatchRole.GOALKEEPER, (-0.95, 0.0, 0.0)),
    (MatchRole.DEFENDER, (0.50, 0.60, 0.0)),
    (MatchRole.PLAYMAKER, (1.75, -1.10, 0.0)),
    (MatchRole.FINISHER, (2.55, 1.30, 0.0)),
)


def build_four_vs_four_fixture(
    asset_root: Path, *, forward_receiver_lane: bool = False
) -> IndependentTeamFixture:
    if not isinstance(forward_receiver_lane, bool):
        raise ValueError("formation selector must be boolean")
    foundation = build_independent_three_vs_three_fixture(asset_root)
    ids = {t: tuple(f"{t}.{role.value}" for role, _ in FORMATION) for t in ("red", "blue")}
    cells, players = [], []
    for team in ("red", "blue"):
        for role, home in FORMATION:
            if forward_receiver_lane and role is MatchRole.FINISHER:
                home = (3.50, -1.10, 0.0)
            origin = home if team == "red" else (6.0 - home[0], -home[1], home[2])
            agent_id = f"{team}.{role.value}"
            cell = build_independent_agent_cell(
                agent_id=agent_id,
                team_id=team,
                primary_role=role,
                teammate_ids=tuple(i for i in ids[team] if i != agent_id),
                opponent_ids=ids["blue" if team == "red" else "red"],
                body_hash=foundation.cells[0].growth_scope.body_hash,
                foundation_policy_hash=foundation.foundation_policy_hash,
                home_position_m=origin,
            )
            cells.append(
                replace(
                    cell,
                    tactical_profile=replace(
                        cell.tactical_profile,
                        active_competition=True,
                    ),
                )
            )
            players.append(
                G1PitchPlayerSpec(
                    agent_id=agent_id,
                    body_prefix=""
                    if agent_id == "red.goalkeeper"
                    else agent_id.replace(".", "_") + "_",
                    origin_m=origin,
                    yaw_rad=0.0 if team == "red" else math.pi,
                    goalkeeper_gloves=role is MatchRole.GOALKEEPER,
                )
            )
    return IndependentTeamFixture(
        roster=TeamRoleRoster("s212.symmetric.4v4", tuple(c.self_model for c in cells)),
        cells=tuple(cells),
        players=tuple(players),
        goal=foundation.goal,
        foundation_policy_hash=foundation.foundation_policy_hash,
    )
