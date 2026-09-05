"""S200 evidence loop for causally routed PASS and SHOOT contact options."""

from __future__ import annotations

import argparse
import json
import tempfile
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from rosclaw_soccer.growth.physical_option_router import PhysicalSoccerOption
from rosclaw_soccer.sim.contracts import ShotParameters, hash_bytes, hash_json
from rosclaw_soccer.skills.team.independent_option_world import (
    IndependentOptionScenario,
    IndependentOptionWorldConfig,
    simulate_independent_physical_option,
)
from rosclaw_soccer.training.independent_team_growth import (
    IndependentTeamFixture,
    build_independent_three_vs_three_fixture,
)


def default_independent_option_scenarios() -> tuple[IndependentOptionScenario, ...]:
    return (
        IndependentOptionScenario(
            scenario_id="s200.physical.red-pass",
            option_agent_id="red.playmaker",
            expected_option=PhysicalSoccerOption.PASS,
            ball_initial_position_m=(1.25, 0.0, 0.115),
            ball_initial_velocity_mps=(0.0, 0.0, 0.0),
            preferred_target_m=(3.55, 0.15, 0.115),
            parameters=ShotParameters(
                swing_amplitude=0.88,
                foot_yaw_offset=0.04,
                foot_pitch_offset=0.02,
                recovery_step_length=0.06,
            ),
            seed=200_600,
        ),
        IndependentOptionScenario(
            scenario_id="s200.physical.red-directed-shot",
            option_agent_id="red.finisher",
            expected_option=PhysicalSoccerOption.SHOOT,
            ball_initial_position_m=(1.25, 0.0, 0.115),
            ball_initial_velocity_mps=(0.0, 0.0, 0.0),
            preferred_target_m=(7.50, 0.80, 1.50),
            parameters=ShotParameters(
                swing_amplitude=1.00,
                foot_yaw_offset=0.085,
                foot_pitch_offset=0.01,
                recovery_step_length=0.04,
            ),
            seed=200_601,
        ),
    )


def build_physical_option_fixture(
    base: IndependentTeamFixture,
    scenario: IndependentOptionScenario,
) -> IndependentTeamFixture:
    """Move only initial bodies so the authorized option owns canonical policy space."""

    positions = {
        "red.goalkeeper": (-1.15, -1.40, 0.0),
        "red.playmaker": (2.70, -1.90, 0.0),
        "red.finisher": (3.55, 0.15, 0.0),
        "blue.goalkeeper": (7.10, 0.00, 0.0),
        "blue.playmaker": (5.00, 1.80, 0.0),
        "blue.finisher": (4.10, -1.80, 0.0),
    }
    positions[scenario.option_agent_id] = (0.0, 0.0, 0.0)
    prefix_by_id = {player.agent_id: player.body_prefix for player in base.players}
    previous_source = next(agent_id for agent_id, prefix in prefix_by_id.items() if prefix == "")
    prefix_by_id[previous_source] = previous_source.replace(".", "_") + "_"
    prefix_by_id[scenario.option_agent_id] = ""
    players = tuple(
        replace(
            player,
            body_prefix=prefix_by_id[player.agent_id],
            origin_m=positions[player.agent_id],
            yaw_rad=0.0 if player.agent_id.startswith("red.") else np.pi,
        )
        for player in base.players
    )
    return replace(base, players=players)


def run_independent_option_growth(
    *,
    evidence_dir: Path,
    asset_root: Path,
    config: IndependentOptionWorldConfig | None = None,
) -> dict[str, Any]:
    root = evidence_dir.expanduser().resolve()
    if root.exists() and any(root.iterdir()):
        raise ValueError("independent-option evidence directory must be empty")
    root.mkdir(parents=True, exist_ok=True)
    active = config or IndependentOptionWorldConfig()
    base = build_independent_three_vs_three_fixture(asset_root)
    rows: list[dict[str, Any]] = []
    for index, scenario in enumerate(default_independent_option_scenarios()):
        fixture = build_physical_option_fixture(base, scenario)
        case_dir = root / f"case-{index:03d}"
        case_dir.mkdir()
        primary, primary_trace = simulate_independent_physical_option(
            asset_root=asset_root,
            roster=fixture.roster,
            cells=fixture.cells,
            players=fixture.players,
            scenario=scenario,
            goal=fixture.goal,
            config=active,
        )
        replay, replay_trace = simulate_independent_physical_option(
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
                "scenario": _scenario_dict(scenario),
                "scenario_hash": scenario.scenario_hash,
                "fixture_hash": fixture.fixture_hash,
                "result": primary.to_dict(),
                "result_hash": primary.result_hash,
                "primary_artifact": primary_artifact,
                "replay_artifact": replay_artifact,
                "exact_replay": exact,
            }
        )
    results = [cast(dict[str, Any], row["result"]) for row in rows]
    outcomes = [cast(dict[str, Any], value["outcome"]) for value in results]
    gates = {
        "pass_and_shoot_authorized": {
            cast(dict[str, Any], value["request"])["option"] for value in results
        }
        == {"pass", "shoot"},
        "physical_contact_both": all(
            value["physical_contact_success"] is True for value in outcomes
        ),
        "all_six_cells_decide": all(
            value["all_cells_decided_each_frame"] is True for value in results
        ),
        "all_six_g1_present": all(
            value["all_cells_physically_present"] is True for value in results
        ),
        "no_post_start_state_write": all(
            value["root_pose_write_after_start"] is False
            and value["ball_state_write_after_start"] is False
            for value in outcomes
        ),
        "safe_both": all(value["safe"] is True for value in outcomes),
        "exact_replay_both": all(row["exact_replay"] is True for row in rows),
    }
    passed = all(gates.values())
    report: dict[str, Any] = {
        "schema_version": "rosclaw_soccer.independent_option_growth_evidence.v1",
        "status": "PASS_CAUSAL_PHYSICAL_OPTIONS" if passed else "REJECTED_PHYSICAL_OPTIONS",
        "passed": passed,
        "base_fixture_hash": base.fixture_hash,
        "config": asdict(active),
        "config_hash": active.config_hash,
        "rows": rows,
        "gates": gates,
        "evidence_boundary": {
            "physics_authority": "CPU_MUJOCO",
            "activation_ceiling": "SIM_ONLY",
            "whole_body_g1_count": 6,
            "one_agent_cell_per_body": True,
            "decentralized_execution": True,
            "physical_pass_contact_routed": True,
            "physical_shoot_contact_routed": True,
            "continuous_pass_receive_shoot_chain_complete": False,
            "physical_save_router_complete": False,
            "pixels_used_for_scoring": False,
            "hardware_command_sent": False,
        },
        "implementation_hash": hash_json(
            {
                "growth": hash_bytes(Path(__file__).read_bytes()),
                "router": hash_bytes(
                    (Path(__file__).parents[1] / "growth/physical_option_router.py").read_bytes()
                ),
                "world": hash_bytes(
                    (
                        Path(__file__).parents[1] / "skills/team/independent_option_world.py"
                    ).read_bytes()
                ),
            }
        ),
    }
    report["report_hash"] = hash_json(report)
    _atomic_json(root / "physical-option-exam.json", report)
    return validate_independent_option_growth(root / "physical-option-exam.json")


def validate_independent_option_growth(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    value = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("independent-option evidence must be an object")
    expected = value.pop("report_hash", None)
    try:
        rows = value.get("rows")
        if not isinstance(rows, list) or len(rows) != 2:
            raise ValueError("independent-option rows are incomplete")
        for index, row in enumerate(rows):
            if not isinstance(row, dict):
                raise ValueError("independent-option row is malformed")
            for label in ("primary_artifact", "replay_artifact"):
                artifact = row.get(label)
                if not isinstance(artifact, dict):
                    raise ValueError("independent-option artifact binding is absent")
                artifact_path = resolved.parent / f"case-{index:03d}" / str(artifact.get("file"))
                if not artifact_path.is_file() or hash_bytes(
                    artifact_path.read_bytes()
                ) != artifact.get("file_hash"):
                    raise ValueError("independent-option trajectory binding changed")
        gates = value.get("gates")
        passed = isinstance(gates, dict) and all(item is True for item in gates.values())
        boundary = value.get("evidence_boundary")
        if (
            expected != hash_json(value)
            or value.get("schema_version") != "rosclaw_soccer.independent_option_growth_evidence.v1"
            or value.get("passed") is not passed
            or value.get("status")
            != ("PASS_CAUSAL_PHYSICAL_OPTIONS" if passed else "REJECTED_PHYSICAL_OPTIONS")
            or not isinstance(boundary, dict)
            or boundary.get("activation_ceiling") != "SIM_ONLY"
            or boundary.get("hardware_command_sent") is not False
            or boundary.get("continuous_pass_receive_shoot_chain_complete") is not False
        ):
            raise ValueError("independent-option integrity or authority contract is invalid")
    finally:
        if expected is not None:
            value["report_hash"] = expected
    return cast(dict[str, Any], value)


def _scenario_dict(value: IndependentOptionScenario) -> dict[str, Any]:
    result = asdict(value)
    result["expected_option"] = value.expected_option.value
    return result


def _write_trajectory(path: Path, trajectory: dict[str, NDArray[Any]]) -> dict[str, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as descriptor:
        temporary = Path(descriptor.name)
    try:
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **trajectory)  # type: ignore[arg-type]
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "file": path.name,
        "file_hash": hash_bytes(path.read_bytes()),
        "trajectory_digest": str(
            hash_json({name: np.asarray(values).tolist() for name, values in trajectory.items()})
        ),
    }


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, suffix=".json", delete=False
    ) as descriptor:
        temporary = Path(descriptor.name)
        json.dump(value, descriptor, indent=2, sort_keys=True)
        descriptor.write("\n")
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--validate", type=Path)
    args = parser.parse_args()
    if args.validate is not None:
        report = validate_independent_option_growth(args.validate)
    else:
        report = run_independent_option_growth(
            evidence_dir=args.evidence_dir,
            asset_root=args.asset_root,
        )
    print(json.dumps({"status": report["status"], "report_hash": report["report_hash"]}))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "build_physical_option_fixture",
    "default_independent_option_scenarios",
    "run_independent_option_growth",
    "validate_independent_option_growth",
]
