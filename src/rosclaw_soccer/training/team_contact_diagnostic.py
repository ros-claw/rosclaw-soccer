"""Read-only torque attribution in an actual shared eight-player episode.

An intent is not a pass. This diagnostic records physical contacts and opposing
motor contributions without changing the world or treating motion as success.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.growth.locomotion_contact_teacher import G1LocomotionContactTeacherConfig
from rosclaw_soccer.sim.contracts import hash_bytes
from rosclaw_soccer.skills.team.independent_team_world import (
    IndependentTeamWorldConfig,
    IndependentTeamWorldScenario,
    simulate_independent_team_world,
)
from rosclaw_soccer.training.four_vs_four_match import build_four_vs_four_fixture
from rosclaw_soccer.training.shared_keeper_reach_exam import _source_hash
from rosclaw_soccer.world.goalkeeper_glove_material import GoalkeeperGloveMaterial


def summarize(trace: dict[str, Any]) -> dict[str, Any]:
    residual = np.asarray(trace["contact_teacher_residual_torque_nm"])
    baseline = np.asarray(trace["contact_teacher_baseline_torque_nm"])
    guarded = np.asarray(trace["contact_teacher_guarded_torque_nm"])
    valid = np.asarray(trace["contact_teacher_last_substep_valid"], dtype=bool)
    norm = np.linalg.norm(residual, axis=1)
    active = valid & (norm > 1)
    projection = np.sum(baseline * residual, axis=1) / np.maximum(norm, 1e-9)
    retained = np.sum(guarded * residual, axis=1) / np.maximum(norm, 1e-9)
    contact = np.asarray(trace["ball_contact_force_n"]) > 1
    foot = np.isin(trace["ball_contact_effector_code"], (1, 2))
    return dict(
        teacher_measured_frames=int(active.sum()),
        baseline_opposes_teacher_fraction=float(np.mean(projection[active] < 0))
        if active.any()
        else None,
        median_baseline_projection_nm=float(np.median(projection[active]))
        if active.any()
        else None,
        median_teacher_norm_nm=float(np.median(norm[active])) if active.any() else None,
        median_guarded_projection_nm=float(np.median(retained[active])) if active.any() else None,
        physical_foot_contact_frames=int(np.sum(contact & foot)),
        physical_foot_contact_agent_codes=sorted(
            int(v) for v in np.unique(np.asarray(trace["ball_contact_agent_code"])[contact & foot])
        ),
        note="50 Hz samples of final 2 ms control substep; frames are not completed passes",
    )


def run(
    root: Path,
    baseline_path: Path,
    output: Path,
    *,
    stiffness_scale: float = 1.0,
    guard_margin: float = 0.04,
) -> dict[str, Any]:
    source = _source_hash()
    baseline_raw = baseline_path.read_bytes()
    baseline = json.loads(baseline_raw)
    fixture = build_four_vs_four_fixture(root, basic_ball_play=True)
    if baseline["fixture_hash"] != fixture.fixture_hash:
        raise ValueError("diagnostic fixture differs from the baseline")
    config = dict(baseline["config"])
    config["joint_guard_margin_rad"] = guard_margin
    if config.get("keeper_reach") is not None or config.get("owned_contact_policy") is not None:
        raise ValueError("diagnostic baseline requires plain frozen locomotion")
    if config.get("glove_material") is not None:
        config["glove_material"] = GoalkeeperGloveMaterial(**config["glove_material"])
    teacher = G1LocomotionContactTeacherConfig(
        **dict(baseline["teacher"], contact_leg_stiffness_scale=stiffness_scale)
    )
    active = IndependentTeamWorldConfig(**config)
    output.mkdir(parents=True, exist_ok=False)
    result, trace = simulate_independent_team_world(
        asset_root=root,
        roster=fixture.roster,
        cells=fixture.cells,
        players=fixture.players,
        goal=fixture.goal,
        scenario=IndependentTeamWorldScenario(**baseline["scenario"]),
        config=active,
        contact_teacher_config=teacher,
    )
    arrays: dict[str, Any] = dict(trace)
    np.savez_compressed(output / "trajectory.npz", **arrays)
    report = dict(
        result=result.to_dict(),
        diagnostic=summarize(trace),
        baseline_hash=str(hash_bytes(baseline_raw)),
        fixture_hash=fixture.fixture_hash,
        source_hash=source,
        source_integrity_verified=source == _source_hash(),
        trajectory_hash=str(hash_bytes((output / "trajectory.npz").read_bytes())),
        config=asdict(active),
        scenario=baseline["scenario"],
        teacher=asdict(teacher),
        activation_ceiling="SIM_ONLY",
        candidate_promoted=False,
    )
    (output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stiffness-scale", type=float, default=1.0)
    parser.add_argument("--guard-margin", type=float, default=0.04)
    args = parser.parse_args()
    report = run(
        args.asset_root,
        args.baseline,
        args.output,
        stiffness_scale=args.stiffness_scale,
        guard_margin=args.guard_margin,
    )
    print(json.dumps(report["diagnostic"], indent=2))


if __name__ == "__main__":
    main()
