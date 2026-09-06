"""Evidence loop for six independently scoped ROSClaw soccer agents.

This stage proves decentralized role decisions, explicit team coordination,
and stable multi-G1 locomotion.  It deliberately does not claim that PASS,
SHOOT, or SAVE intentions are already wired to the mature contact options;
that physical option router is the next stage.  Evidence is CPU MuJoCo and
SIM_ONLY throughout.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from rosclaw_soccer.growth.independent_agent_cell import (
    RosclawSoccerAgentCell,
    build_agent_plasticity_lease,
    build_independent_agent_cell,
)
from rosclaw_soccer.growth.role_self_model import MatchRole, TeamRoleRoster
from rosclaw_soccer.providers.g1.asset_qualification import qualify_g1_assets
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.skills.team.independent_team_world import (
    IndependentTeamWorldConfig,
    IndependentTeamWorldScenario,
    simulate_independent_team_world,
)
from rosclaw_soccer.world.field import G1TrainingGoalSpec
from rosclaw_soccer.world.multi_player import G1PitchPlayerSpec

_LOCO_POLICY_REL = Path("policy/loco_mode/model/policy_29dof.pt")
_LOCO_CONFIG_REL = Path("policy/loco_mode/config/LocoMode.yaml")
_LOCO_CODE_REL = Path("policy/loco_mode/LocoMode.py")


@dataclass(frozen=True)
class IndependentTeamFixture:
    roster: TeamRoleRoster
    cells: tuple[RosclawSoccerAgentCell, ...]
    players: tuple[G1PitchPlayerSpec, ...]
    goal: G1TrainingGoalSpec
    foundation_policy_hash: str
    schema_version: str = "rosclaw_soccer.independent_team_fixture.v1"

    def __post_init__(self) -> None:
        roster_ids = {agent.agent_id for agent in self.roster.agents}
        if (
            len(roster_ids) != 6
            or {cell.agent_id for cell in self.cells} != roster_ids
            or {player.agent_id for player in self.players} != roster_ids
            or len({cell.cell_hash for cell in self.cells}) != 6
        ):
            raise ValueError("independent 3v3 fixture identities are incomplete")
        for team_id in ("red", "blue"):
            roles = {
                cell.self_model.primary_role
                for cell in self.cells
                if cell.self_model.team_id == team_id
            }
            if roles != {MatchRole.GOALKEEPER, MatchRole.PLAYMAKER, MatchRole.FINISHER}:
                raise ValueError("each 3v3 side requires goalkeeper, playmaker, and finisher")

    @property
    def fixture_hash(self) -> str:
        return str(hash_json(self.to_dict()))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "roster_hash": self.roster.roster_hash,
            "cell_hashes": {cell.agent_id: cell.cell_hash for cell in self.cells},
            "growth_scope_hashes": {
                cell.agent_id: cell.growth_scope.scope_hash for cell in self.cells
            },
            "personal_memory_namespaces": {
                cell.agent_id: cell.growth_scope.personal_memory_namespace for cell in self.cells
            },
            "failure_memory_namespaces": {
                cell.agent_id: cell.growth_scope.failure_memory_namespace for cell in self.cells
            },
            "player_specs": [asdict(player) for player in self.players],
            "goal": asdict(self.goal),
            "foundation_policy_hash": self.foundation_policy_hash,
        }


_LAYOUT = (
    ("red.goalkeeper", "red", MatchRole.GOALKEEPER, (0.20, 0.00, 0.0), "", 0.0),
    (
        "red.playmaker",
        "red",
        MatchRole.PLAYMAKER,
        (2.00, -0.70, 0.0),
        "red_playmaker_",
        0.0,
    ),
    (
        "red.finisher",
        "red",
        MatchRole.FINISHER,
        (3.30, 1.10, 0.0),
        "red_finisher_",
        0.0,
    ),
    (
        "blue.goalkeeper",
        "blue",
        MatchRole.GOALKEEPER,
        (7.10, 0.00, 0.0),
        "blue_goalkeeper_",
        math.pi,
    ),
    (
        "blue.playmaker",
        "blue",
        MatchRole.PLAYMAKER,
        (5.30, -1.10, 0.0),
        "blue_playmaker_",
        math.pi,
    ),
    (
        "blue.finisher",
        "blue",
        MatchRole.FINISHER,
        (4.20, 1.10, 0.0),
        "blue_finisher_",
        math.pi,
    ),
)


def build_independent_three_vs_three_fixture(asset_root: Path) -> IndependentTeamFixture:
    """Create six private growth scopes around one qualified locomotion foundation."""

    qualification = qualify_g1_assets(asset_root)
    qualification.require_eligible()
    locomotion_paths = tuple(
        qualification.asset_root / relative
        for relative in (_LOCO_POLICY_REL, _LOCO_CONFIG_REL, _LOCO_CODE_REL)
    )
    if any(not path.is_file() for path in locomotion_paths):
        raise ValueError("qualified asset checkout lacks the executed locomotion foundation")
    foundation = str(
        hash_json(
            {
                "foundation": "robonaldo.locomotion.frozen",
                "body_hash": qualification.body_hash,
                "artifacts": {
                    str(path.relative_to(qualification.asset_root)): hash_bytes(path.read_bytes())
                    for path in locomotion_paths
                },
            }
        )
    )
    team_ids = {
        team: tuple(agent_id for agent_id, value, _, _, _, _ in _LAYOUT if value == team)
        for team in ("red", "blue")
    }
    cells = tuple(
        build_independent_agent_cell(
            agent_id=agent_id,
            team_id=team,
            primary_role=role,
            teammate_ids=tuple(value for value in team_ids[team] if value != agent_id),
            opponent_ids=team_ids["blue" if team == "red" else "red"],
            body_hash=qualification.body_hash,
            foundation_policy_hash=foundation,
            home_position_m=origin,
        )
        for agent_id, team, role, origin, _, _ in _LAYOUT
    )
    players = tuple(
        G1PitchPlayerSpec(
            agent_id=agent_id,
            body_prefix=prefix,
            origin_m=origin,
            yaw_rad=yaw,
            goalkeeper_gloves=role is MatchRole.GOALKEEPER,
        )
        for agent_id, _, role, origin, prefix, yaw in _LAYOUT
    )
    goal = G1TrainingGoalSpec(
        plane_x_m=7.50,
        width_m=3.0,
        height_m=2.0,
        depth_m=1.2,
        target_y_m=0.80,
        target_z_m=1.50,
        precision_radius_m=0.10,
    )
    return IndependentTeamFixture(
        roster=TeamRoleRoster("s199.independent.3v3", tuple(cell.self_model for cell in cells)),
        cells=cells,
        players=players,
        goal=goal,
        foundation_policy_hash=foundation,
    )


def default_independent_team_retention_cases() -> tuple[IndependentTeamWorldScenario, ...]:
    """Fresh cases covering pass, bilateral goalkeeping, finish, and distribution."""

    return (
        IndependentTeamWorldScenario(
            scenario_id="s199.retention.red-transition",
            ball_initial_position_m=(2.15, -0.65, 0.115),
            ball_initial_velocity_mps=(0.65, 0.12, 0.0),
            seed=199_500,
        ),
        IndependentTeamWorldScenario(
            scenario_id="s199.retention.blue-attack-red-save",
            ball_initial_position_m=(1.45, 0.40, 0.115),
            ball_initial_velocity_mps=(-1.10, -0.08, 0.0),
            seed=199_501,
        ),
        IndependentTeamWorldScenario(
            scenario_id="s199.retention.red-finish",
            # Give the red finisher possession without spawning the sphere
            # between its feet; the option router must approach from a legal
            # preparation pocket rather than rely on an initial overlap.
            ball_initial_position_m=(3.70, 1.60, 0.115),
            ball_initial_velocity_mps=(0.08, 0.0, 0.0),
            seed=199_502,
        ),
        IndependentTeamWorldScenario(
            scenario_id="s199.retention.blue-distribution",
            ball_initial_position_m=(7.02, 0.05, 0.115),
            ball_initial_velocity_mps=(0.0, 0.0, 0.0),
            seed=199_503,
        ),
    )


def run_independent_team_growth(
    *,
    evidence_dir: Path,
    asset_root: Path,
    config: IndependentTeamWorldConfig | None = None,
) -> dict[str, Any]:
    """Run deterministic replay evidence for the independent 3v3 foundation."""

    root = evidence_dir.expanduser().resolve()
    if root.exists() and any(root.iterdir()):
        raise ValueError("independent-team evidence directory must be empty")
    root.mkdir(parents=True, exist_ok=True)
    active = config or IndependentTeamWorldConfig(simulation_duration_sec=5.0)
    fixture = build_independent_three_vs_three_fixture(asset_root)
    rows: list[dict[str, Any]] = []
    for index, scenario in enumerate(default_independent_team_retention_cases()):
        case_dir = root / f"case-{index:03d}"
        case_dir.mkdir()
        primary, primary_trace = simulate_independent_team_world(
            asset_root=asset_root,
            roster=fixture.roster,
            cells=fixture.cells,
            players=fixture.players,
            scenario=scenario,
            goal=fixture.goal,
            config=active,
        )
        replay, replay_trace = simulate_independent_team_world(
            asset_root=asset_root,
            roster=fixture.roster,
            cells=fixture.cells,
            players=fixture.players,
            scenario=scenario,
            goal=fixture.goal,
            config=active,
        )
        primary_artifact = _write_trajectory(case_dir / "primary.npz", primary_trace)
        replay_artifact = _write_trajectory(case_dir / "replay.npz", replay_trace)
        exact = bool(
            primary.to_dict() == replay.to_dict()
            and primary.trajectory_hash == replay.trajectory_hash
            and primary_artifact["trajectory_digest"] == replay_artifact["trajectory_digest"]
        )
        rows.append(
            {
                "scenario": asdict(scenario),
                "scenario_hash": scenario.scenario_hash,
                "result": primary.to_dict(),
                "result_hash": primary.result_hash,
                "primary_artifact": primary_artifact,
                "replay_artifact": replay_artifact,
                "exact_replay": exact,
            }
        )
    dataset_hash = str(hash_json({"rows": [row["result_hash"] for row in rows]}))
    scenario_contract_hash = str(hash_json({"scenarios": [row["scenario_hash"] for row in rows]}))
    leases = {
        cell.agent_id: build_agent_plasticity_lease(
            cells=fixture.cells,
            focal_agent_id=cell.agent_id,
            dataset_manifest_hash=dataset_hash,
            scenario_contract_hash=scenario_contract_hash,
            maximum_optimizer_steps=2_000,
        ).to_dict()
        for cell in fixture.cells
    }
    results = [cast(dict[str, Any], row["result"]) for row in rows]
    gates = {
        "six_independent_cells": len({cell.cell_hash for cell in fixture.cells}) == 6,
        "six_private_personal_memories": len(
            {cell.growth_scope.personal_memory_namespace for cell in fixture.cells}
        )
        == 6,
        "six_private_failure_memories": len(
            {cell.growth_scope.failure_memory_namespace for cell in fixture.cells}
        )
        == 6,
        "one_plasticity_lease_per_cell": len(leases) == 6
        and all(
            sum(binding["mode"] == "PLASTIC" for binding in value["bindings"]) == 1
            for value in leases.values()
        ),
        "role_complete_red_and_blue": all(
            {
                cell.self_model.primary_role
                for cell in fixture.cells
                if cell.self_model.team_id == team
            }
            == {MatchRole.GOALKEEPER, MatchRole.PLAYMAKER, MatchRole.FINISHER}
            for team in ("red", "blue")
        ),
        "all_retention_worlds_pass": all(value["passed"] is True for value in results),
        "exact_replay_all": all(row["exact_replay"] is True for row in rows),
        "explicit_pass_negotiation_observed": sum(
            int(value["pass_handshake_count"]) for value in results
        )
        > 0,
        "goalkeeper_save_intent_observed": sum(int(value["save_intent_count"]) for value in results)
        > 0,
        "finisher_shot_intent_observed": sum(int(value["shot_intent_count"]) for value in results)
        > 0,
        "goalkeeper_distribution_intent_observed": sum(
            int(value["distribution_intent_count"]) for value in results
        )
        > 0,
        "both_goalkeepers_defend": all(
            any(
                quality["agent_id"] == agent_id and int(quality["save_intent_count"]) > 0
                for result in results
                for quality in result["qualities"]
            )
            for agent_id in ("red.goalkeeper", "blue.goalkeeper")
        ),
        "no_robot_robot_contact": all(
            int(value["robot_robot_contact_count"]) == 0 for value in results
        ),
    }
    passed = all(gates.values())
    report: dict[str, Any] = {
        "schema_version": "rosclaw_soccer.independent_team_growth_evidence.v1",
        "status": "PASS_INDEPENDENT_3V3_FOUNDATION" if passed else "REJECTED_3V3_FOUNDATION",
        "passed": passed,
        "fixture": fixture.to_dict(),
        "fixture_hash": fixture.fixture_hash,
        "config": asdict(active),
        "config_hash": active.config_hash,
        "rows": rows,
        "plasticity_leases": leases,
        "gates": gates,
        "evidence_boundary": {
            "physics_authority": "CPU_MUJOCO",
            "activation_ceiling": "SIM_ONLY",
            "whole_body_g1_count": 6,
            "red_agent_count": 3,
            "blue_agent_count": 3,
            "one_agent_cell_per_body": True,
            "decentralized_execution": True,
            "explicit_team_coordination": True,
            "movement_executed_by_frozen_neural_locomotion": True,
            "tactical_policy_kind": "CAUSAL_WARM_START_TEACHER",
            "contact_skill_router_complete": False,
            "pixels_used_for_scoring": False,
            "hardware_command_sent": False,
        },
        "implementation_hash": hash_json(
            {
                "growth": hash_bytes(Path(__file__).read_bytes()),
                "cells": hash_bytes(
                    (Path(__file__).parents[1] / "growth/independent_agent_cell.py").read_bytes()
                ),
                "world": hash_bytes(
                    (
                        Path(__file__).parents[1] / "skills/team/independent_team_world.py"
                    ).read_bytes()
                ),
                "multi_player_world": hash_bytes(
                    (Path(__file__).parents[1] / "world/multi_player.py").read_bytes()
                ),
                "base_field": hash_bytes(
                    (Path(__file__).parents[1] / "world/field.py").read_bytes()
                ),
            }
        ),
    }
    report["report_hash"] = hash_json(report)
    _atomic_json(root / "retention-exam.json", report)
    return validate_independent_team_growth(root / "retention-exam.json")


def validate_independent_team_growth(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    value = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("independent-team evidence must be an object")
    expected = value.pop("report_hash", None)
    try:
        rows = value.get("rows")
        if not isinstance(rows, list) or len(rows) != 4:
            raise ValueError("independent-team retention rows are incomplete")
        for index, row in enumerate(rows):
            if not isinstance(row, dict):
                raise ValueError("independent-team row is malformed")
            for label in ("primary_artifact", "replay_artifact"):
                artifact = row.get(label)
                if not isinstance(artifact, dict):
                    raise ValueError("independent-team artifact binding is absent")
                artifact_path = resolved.parent / f"case-{index:03d}" / str(artifact.get("file"))
                if not artifact_path.is_file() or hash_bytes(
                    artifact_path.read_bytes()
                ) != artifact.get("file_hash"):
                    raise ValueError("independent-team trajectory binding changed")
        gates = value.get("gates")
        passed = isinstance(gates, dict) and all(item is True for item in gates.values())
        boundary = value.get("evidence_boundary")
        if (
            expected != hash_json(value)
            or value.get("schema_version") != "rosclaw_soccer.independent_team_growth_evidence.v1"
            or value.get("passed") is not passed
            or value.get("status")
            != ("PASS_INDEPENDENT_3V3_FOUNDATION" if passed else "REJECTED_3V3_FOUNDATION")
            or not isinstance(boundary, dict)
            or boundary.get("activation_ceiling") != "SIM_ONLY"
            or boundary.get("hardware_command_sent") is not False
            or boundary.get("contact_skill_router_complete") is not False
        ):
            raise ValueError("independent-team integrity or authority contract is invalid")
    finally:
        if expected is not None:
            value["report_hash"] = expected
    return cast(dict[str, Any], value)


def _write_trajectory(path: Path, trajectory: dict[str, NDArray[Any]]) -> dict[str, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as descriptor:
        temporary = Path(descriptor.name)
    try:
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **trajectory)  # type: ignore[arg-type]
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "file": path.name,
        "file_hash": hash_bytes(path.read_bytes()),
        "trajectory_digest": str(
            hash_json(
                {
                    name: hash_bytes(np.ascontiguousarray(value).tobytes())
                    for name, value in sorted(trajectory.items())
                }
            )
        ),
    }


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, suffix=".json", mode="w", encoding="utf-8", delete=False
    ) as descriptor:
        temporary = Path(descriptor.name)
        try:
            json.dump(value, descriptor, ensure_ascii=False, indent=2, sort_keys=True)
            descriptor.write("\n")
            descriptor.flush()
            os.fsync(descriptor.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", required=True, type=Path)
    parser.add_argument("--asset-root", required=True, type=Path)
    return parser


def main() -> None:
    arguments = _parser().parse_args()
    value = run_independent_team_growth(
        evidence_dir=arguments.evidence_dir,
        asset_root=arguments.asset_root,
    )
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = [
    "IndependentTeamFixture",
    "build_independent_three_vs_three_fixture",
    "default_independent_team_retention_cases",
    "run_independent_team_growth",
    "validate_independent_team_growth",
]
