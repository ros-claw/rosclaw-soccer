"""Paired CPU MuJoCo measurements of purposeful multi-player engagement."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.providers.g1.asset_qualification import trajectory_digest
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.skills.team.independent_team_world import simulate_independent_team_world
from rosclaw_soccer.training.continuous_competitive_match_growth import (
    build_continuous_competitive_fixture,
    default_continuous_match_config,
    default_continuous_match_scenario,
)
from rosclaw_soccer.training.phase_conditioned_strike_growth import (
    default_phase_strike_controller,
    default_phase_strike_option,
    default_phase_strike_teacher,
)


def run_probe(*, asset_root: Path, output: Path, active: bool, duration: float) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(output)
    fixture = build_continuous_competitive_fixture(asset_root)
    cells = tuple(
        replace(cell, tactical_profile=replace(cell.tactical_profile, active_competition=active))
        for cell in fixture.cells
    )
    config = replace(default_continuous_match_config(), simulation_duration_sec=duration)
    if active:
        config = replace(
            config,
            minimum_player_separation_m=1.20,
            collision_avoidance_gain=3.0,
            duel_lateral_offset_m=0.45,
        )
    scenario = default_continuous_match_scenario()
    results, trajectories = [], []
    for _ in range(2):
        result, trajectory = simulate_independent_team_world(
            asset_root=asset_root,
            roster=fixture.roster,
            cells=cells,
            players=fixture.players,
            scenario=scenario,
            goal=fixture.goal,
            config=config,
            contact_teacher_config=default_phase_strike_teacher(),
            option_bridge_config=default_phase_strike_option(),
            strike_phase_config=default_phase_strike_controller(),
        )
        results.append(result.to_dict())
        trajectories.append(trajectory)
    rows = engagement_rows(trajectories[0], tuple(sorted(c.agent_id for c in cells)))
    exact = trajectory_digest(trajectories[0]) == trajectory_digest(trajectories[1])
    output.mkdir(parents=True)
    for name, trajectory in zip(("primary", "replay"), trajectories, strict=True):
        np.savez_compressed(output / f"{name}.npz", **trajectory)  # type: ignore[arg-type]
    report = {
        "schema_version": "rosclaw_soccer.active_team_probe.v1",
        "active_competition": active,
        "world_config": asdict(config),
        "scenario": asdict(scenario),
        "goal": asdict(fixture.goal),
        "cells": [c.to_dict() for c in cells],
        "results": results,
        "engagement": rows,
        "exact_replay": exact,
        "trajectory_digests": [trajectory_digest(t) for t in trajectories],
        "artifacts": {
            name: hash_bytes((output / name).read_bytes()) for name in ("primary.npz", "replay.npz")
        },
        "activation_ceiling": "SIM_ONLY",
        "promotion_eligible": False,
        "hardware_command_sent": False,
    }
    report["report_hash"] = hash_json(report)
    (output / "probe.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def engagement_rows(trajectory: dict[str, Any], agent_ids: tuple[str, ...]) -> list[dict[str, Any]]:
    time = np.asarray(trajectory["time"])
    dt = np.diff(time)
    if len(time) < 2 or np.any(dt <= 0):
        raise ValueError("engagement needs strictly increasing physical time")
    rows = []
    for code, agent_id in enumerate(agent_ids, 1):
        key = agent_id.replace(".", "_")
        xy = np.asarray(trajectory[key + "_pelvis_pose"])[:, :2]
        steps = np.linalg.norm(np.diff(xy, axis=0), axis=1)
        moving = steps / dt > 0.15
        idle = longest = 0.0
        for moving_now, elapsed in zip(moving, dt, strict=True):
            idle = 0.0 if moving_now else idle + float(elapsed)
            longest = max(longest, idle)
        contact = np.asarray(trajectory["ball_contact_agent_code"]) == code
        rows.append(
            {
                "agent_id": agent_id,
                "path_length_m": float(steps.sum()),
                "moving_fraction": float(np.sum(dt[moving]) / dt.sum()),
                "longest_stationary_sec": longest,
                "physical_ball_contact_frames": int(contact.sum()),
            }
        )
    return rows


def validate_probe(path: Path) -> dict[str, Any]:
    report = json.loads(path.read_text())
    if not isinstance(report, dict):
        raise ValueError("active team report must be an object")
    digest = report.pop("report_hash")
    if digest != hash_json(report):
        raise ValueError("active team report hash changed")
    traces = []
    for name in ("primary.npz", "replay.npz"):
        source = path.parent / name
        if hash_bytes(source.read_bytes()) != report["artifacts"][name]:
            raise ValueError("active team trajectory file changed")
        with np.load(source, allow_pickle=False) as archive:
            traces.append({k: archive[k] for k in archive.files})
    digests = [trajectory_digest(t) for t in traces]
    ids = tuple(sorted(c["self_model"]["agent_id"] for c in report["cells"]))
    if (
        report["schema_version"] != "rosclaw_soccer.active_team_probe.v1"
        or report["trajectory_digests"] != digests
        or report["exact_replay"] is not True
        or digests[0] != digests[1]
        or report["results"][0] != report["results"][1]
        or any(r["trajectory_hash"] != d for r, d in zip(report["results"], digests, strict=True))
        or report["engagement"] != engagement_rows(traces[0], ids)
        or report["activation_ceiling"] != "SIM_ONLY"
        or report["hardware_command_sent"] is not False
        or report["promotion_eligible"] is not False
    ):
        raise ValueError("active team evidence failed validation")
    report["report_hash"] = digest
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--active", action="store_true")
    parser.add_argument("--duration", type=float, default=12.0)
    args = parser.parse_args()
    report = run_probe(
        asset_root=args.asset_root, output=args.output, active=args.active, duration=args.duration
    )
    print(json.dumps({k: report[k] for k in ("engagement", "exact_replay", "results")}))


if __name__ == "__main__":
    main()
