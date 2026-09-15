"""Scoped on-policy residual learning on actual 4v4 trajectories.

No reset between events, scripted ball forces, or automatic candidate activation.
Frozen roles use deterministic parent actions during collection as well as exams.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.growth.locomotion_contact_teacher import G1RollingOptionBridgeConfig
from rosclaw_soccer.growth.near_ball_residual import NearBallResidualPolicy
from rosclaw_soccer.growth.owned_ball_contact import OwnedBallContactPolicy
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.skills.team.independent_team_world import (
    IndependentTeamWorldConfig,
    IndependentTeamWorldScenario,
)
from rosclaw_soccer.training.continuous_competitive_match_growth import (
    default_continuous_match_config,
    default_continuous_match_options,
    run_continuous_competitive_match_growth,
    validate_continuous_competitive_match_growth,
)
from rosclaw_soccer.training.four_vs_four_match import build_four_vs_four_fixture
from rosclaw_soccer.training.independent_team_growth import IndependentTeamFixture
from rosclaw_soccer.training.near_ball_plasticity import recorded_training_scope
from rosclaw_soccer.training.near_ball_residual_ppo import update_private_actors


@dataclass(frozen=True)
class MatchCollection:
    assets: Path
    output: Path
    checkpoint: Path
    seed: int
    scope: tuple[str, ...]
    explore: bool = True
    ball_y_m: float = 0.0
    outlet_stance_depth_m: float | None = None
    strict_handoff: bool = False
    finisher_option_learning: bool = False
    ball_x_m: float = 3.0
    prospective_motor: bool = False
    task_context_bound: bool = False
    motor_lateral_limit_m: float = 0.60
    prospective_strike_approach: bool = False
    teammate_approach_clearance_m: float = 0.0
    keeper_distribution_preview: bool = False
    per_player_options: bool = False
    continuous_motor_rearm: bool = False
    receiver_commitment_priority: bool = False


def collection_options(
    world: IndependentTeamWorldConfig,
    *,
    prospective: bool = False,
    bound_context: bool = False,
    lateral_limit_m: float = 0.60,
    per_player_options: bool = False,
    continuous_motor_rearm: bool = False,
) -> G1RollingOptionBridgeConfig:
    return replace(
        default_continuous_match_options(world),
        prospective_enabled=prospective,
        task_context_bound=bound_context,
        maximum_strike_lateral_error_m=lateral_limit_m,
        per_player_options_enabled=per_player_options,
        continuous_rearm_enabled=continuous_motor_rearm,
    )


def training_kickoffs(
    *, varied: bool, balanced_motor: bool, buildup: bool = False
) -> tuple[tuple[float, float], ...]:
    flags = (varied, balanced_motor, buildup)
    if any(type(flag) is not bool for flag in flags) or sum(flags) > 1:
        raise ValueError("explicit nonoverlapping kickoff curriculum required")
    if buildup:
        return ((-0.4, 0.0), (6.4, 0.0), (1.05, 0.6), (4.95, -0.6))
    if balanced_motor:
        return ((3.2, -0.3), (2.8, 0.3))
    return tuple((3.0, y) for y in ((0.0, -0.02, 0.02, -0.04, 0.04) if varied else (0.0,)))


def evaluation_kickoffs(*, buildup: bool = False) -> tuple[tuple[str, float, float], ...]:
    if type(buildup) is not bool:
        raise ValueError("evaluation curriculum must be explicit")
    if not buildup:
        return tuple((f"{y:+.2f}", 3.0, y) for y in (-0.06, 0.0, 0.06))
    bases = training_kickoffs(varied=False, balanced_motor=False, buildup=True)
    return (("centre", 3.0, 0.0),) + tuple(
        (f"outlet-{index}-{offset:+.2f}", x, y + offset)
        for index, (x, y) in enumerate(bases)
        for offset in (-0.06, 0.06)
    )


def collection_fixture(assets: Path, *, keeper_preview: bool = False) -> IndependentTeamFixture:
    if type(keeper_preview) is not bool:
        raise ValueError("keeper distribution preview must be explicit")
    fixture = build_four_vs_four_fixture(assets, basic_ball_play=True)
    return replace(
        fixture,
        cells=tuple(
            replace(
                c,
                tactical_profile=replace(
                    c.tactical_profile,
                    anticipatory_contact=True,
                    keeper_distribution_preview=keeper_preview,
                ),
            )
            for c in fixture.cells
        ),
    )


def collection_world(
    depth: float | None,
    strict_handoff: bool,
    finisher_option_learning: bool = False,
    prospective_strike_approach: bool = False,
    teammate_approach_clearance_m: float = 0.0,
    receiver_commitment_priority: bool = False,
) -> IndependentTeamWorldConfig:
    if type(strict_handoff) is not bool:
        raise ValueError("strict handoff must be explicit")
    if type(finisher_option_learning) is not bool:
        raise ValueError("finisher option learning must be explicit")
    return replace(
        default_continuous_match_config(),
        simulation_duration_sec=25.0,
        bilateral_goals=True,
        stationary_ball_acquisition=True,
        predictive_separation=True,
        all_role_clearance=True,
        strike_residual_enabled=True,
        owned_contact_policy=None if depth is None else OwnedBallContactPolicy(depth_m=depth),
        owned_contact_roles=None if depth is None else ("defender", "goalkeeper", "playmaker"),
        strict_receive_handoff=strict_handoff,
        option_only_residual_roles=("finisher",) if finisher_option_learning else None,
        prospective_strike_approach=prospective_strike_approach,
        teammate_approach_clearance_m=teammate_approach_clearance_m,
        receiver_commitment_priority=receiver_commitment_priority,
    )


def collect(job: MatchCollection) -> str:
    fixture = collection_fixture(job.assets, keeper_preview=job.keeper_distribution_preview)
    world = collection_world(
        job.outlet_stance_depth_m,
        job.strict_handoff,
        job.finisher_option_learning,
        job.prospective_strike_approach,
        job.teammate_approach_clearance_m,
        job.receiver_commitment_priority,
    )
    report = run_continuous_competitive_match_growth(
        evidence_dir=job.output,
        asset_root=job.assets,
        fixture=fixture,
        scenario=IndependentTeamWorldScenario(
            scenario_id="s199.continuous.scoped-ppo.central-kickoff",
            ball_initial_position_m=(job.ball_x_m, job.ball_y_m, fixture.goal.ball_radius_m),
            ball_initial_velocity_mps=(0.0, 0.0, 0.0),
            seed=1928,
        ),
        world_config=world,
        option_config=collection_options(
            world,
            prospective=job.prospective_motor,
            bound_context=job.task_context_bound,
            lateral_limit_m=job.motor_lateral_limit_m,
            per_player_options=job.per_player_options,
            continuous_motor_rearm=job.continuous_motor_rearm,
        ),
        near_ball_policy=NearBallResidualPolicy.load(job.checkpoint),
        near_ball_explore=job.explore,
        near_ball_seed=job.seed,
        near_ball_exploration_agent_ids=job.scope if job.explore else None,
    )
    if not report["exact_replay"]:
        raise ValueError("nonreproducible continuous collection rejected")
    return str(job.output / "continuous-match-exam.json")


def train(
    *,
    assets: Path,
    output: Path,
    checkpoint: Path,
    scope: tuple[str, ...],
    iterations: int = 3,
    workers: int = 4,
    outlet_stance_depth_m: float | None = None,
    strict_handoff: bool = False,
    varied_ball_positions: bool = False,
    finisher_option_learning: bool = False,
    reward_shaping: str = "contact_safety_v1",
    prospective_motor: bool = False,
    task_context_bound: bool = False,
    balanced_motor_kickoffs: bool = False,
    motor_lateral_limit_m: float = 0.60,
    prospective_strike_approach: bool = False,
    teammate_approach_clearance_m: float = 0.0,
    buildup_kickoffs: bool = False,
    keeper_distribution_preview: bool = False,
    per_player_options: bool = False,
    continuous_motor_rearm: bool = False,
    receiver_commitment_priority: bool = False,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(output)
    if reward_shaping not in ("contact_safety_v1", "motor_task_contact_v1"):
        raise ValueError("continuous match reward contract is unsupported")
    if type(iterations) is not int or not 1 <= iterations <= 20:
        raise ValueError("bounded continuous training iterations required")
    if type(workers) is not int or not 1 <= workers <= 4:
        raise ValueError("bounded collection workers required")
    if type(varied_ball_positions) is not bool:
        raise ValueError("ball-position curriculum must be explicit")
    world = collection_world(
        outlet_stance_depth_m,
        strict_handoff,
        finisher_option_learning,
        prospective_strike_approach,
        teammate_approach_clearance_m,
        receiver_commitment_priority,
    )
    if prospective_strike_approach and not (prospective_motor and task_context_bound):
        raise ValueError("prospective approach requires a bound prospective motor")
    kickoffs = training_kickoffs(
        varied=varied_ball_positions,
        balanced_motor=balanced_motor_kickoffs,
        buildup=buildup_kickoffs,
    )
    exams = evaluation_kickoffs(buildup=buildup_kickoffs)
    if type(keeper_distribution_preview) is not bool:
        raise ValueError("keeper distribution preview must be explicit")
    options = collection_options(
        world,
        prospective=prospective_motor,
        bound_context=task_context_bound,
        lateral_limit_m=motor_lateral_limit_m,
        per_player_options=per_player_options,
        continuous_motor_rearm=continuous_motor_rearm,
    )
    if balanced_motor_kickoffs and not (prospective_motor and finisher_option_learning):
        raise ValueError("balanced motor curriculum requires prospective finisher learning")
    parent = NearBallResidualPolicy.load(checkpoint)
    if type(scope) is not tuple or recorded_training_scope(list(scope), parent.agent_ids) != scope:
        raise ValueError("explicit canonical trainable scope required")
    if finisher_option_learning and not set(scope).issubset(("red.finisher", "blue.finisher")):
        raise ValueError("finisher option training must freeze all other private heads")
    fixture_hash = collection_fixture(
        assets, keeper_preview=keeper_distribution_preview
    ).fixture_hash
    initial = parent
    output.mkdir(parents=True)
    parent.save(output / f"generation-{parent.generation:03d}.npz")
    manifest: dict[str, Any] = {
        "schema": "rosclaw_soccer.continuous_match_residual_ppo.v1",
        "activation_ceiling": "SIM_ONLY",
        "promotion_eligible": False,
        "initial_policy_hash": parent.policy_hash,
        "initial_checkpoint_file_hash": hash_bytes(checkpoint.read_bytes()),
        "source_hash": hash_bytes(Path(__file__).read_bytes()),
        "trainable_agent_ids": list(scope),
        "exploration_agent_ids": list(scope),
        "training_ball_y_m": [y for _, y in kickoffs],
        "training_ball_xy_m": [list(xy) for xy in kickoffs],
        "varied_ball_positions": varied_ball_positions,
        "balanced_motor_kickoffs": balanced_motor_kickoffs,
        "buildup_kickoffs": buildup_kickoffs,
        "keeper_distribution_preview": keeper_distribution_preview,
        "per_player_options": per_player_options,
        "continuous_motor_rearm": continuous_motor_rearm,
        "receiver_commitment_priority": receiver_commitment_priority,
        "collection_fixture_hash": fixture_hash,
        "collection_option_config_hash": options.config_hash,
        "prospective_motor": prospective_motor,
        "task_context_bound": task_context_bound,
        "motor_lateral_limit_m": motor_lateral_limit_m,
        "prospective_strike_approach": prospective_strike_approach,
        "teammate_approach_clearance_m": teammate_approach_clearance_m,
        "collection_world_config_hash": world.config_hash,
        "outlet_stance_depth_m": outlet_stance_depth_m,
        "strict_handoff": strict_handoff,
        "finisher_option_learning": finisher_option_learning,
        "evaluation_ball_y_m": [y for _, _, y in exams],
        "evaluation_ball_xy_m": [[x, y] for _, x, y in exams],
        "evaluation_case_names": [name for name, _, _ in exams],
        "optimizer_epochs": 8,
        "gamma": 0.997,
        "trace_decay": 0.997,
        "reward_shaping": reward_shaping,
        "iterations": [],
        "evaluation": [],
    }
    for iteration in range(iterations):
        jobs = [
            MatchCollection(
                assets,
                output / f"train-{iteration:03d}-{index}",
                output / f"generation-{parent.generation:03d}.npz",
                196000 + 4 * iteration + index,
                scope,
                ball_x_m=kickoffs[(iteration * 4 + index) % len(kickoffs)][0],
                ball_y_m=kickoffs[(iteration * 4 + index) % len(kickoffs)][1],
                prospective_motor=prospective_motor,
                task_context_bound=task_context_bound,
                motor_lateral_limit_m=motor_lateral_limit_m,
                prospective_strike_approach=prospective_strike_approach,
                teammate_approach_clearance_m=teammate_approach_clearance_m,
                outlet_stance_depth_m=outlet_stance_depth_m,
                strict_handoff=strict_handoff,
                finisher_option_learning=finisher_option_learning,
                keeper_distribution_preview=keeper_distribution_preview,
                per_player_options=per_player_options,
                continuous_motor_rearm=continuous_motor_rearm,
                receiver_commitment_priority=receiver_commitment_priority,
            )
            for index in range(4)
        ]
        with ProcessPoolExecutor(
            max_workers=workers, mp_context=multiprocessing.get_context("spawn")
        ) as pool:
            paths = list(pool.map(collect, jobs))
        traces, hashes = [], []
        for path in paths:
            report = validate_continuous_competitive_match_growth(Path(path))
            if report["near_ball_policy_hash"] != parent.policy_hash:
                raise ValueError("collection parent binding differs")
            if report["sampling"]["exploration_agent_ids"] != list(scope):
                raise ValueError("collection exploration scope differs")
            with np.load(Path(path).parent / "primary.npz", allow_pickle=False) as archive:
                traces.append({k: archive[k] for k in archive.files})
            hashes.append(report["report_hash"])
        child, rows = update_private_actors(
            parent,
            traces,
            epochs=8,
            gamma=0.997,
            trace_decay=0.997,
            reward_shaping=reward_shaping,
            trainable_agent_ids=scope,
        )
        child.save(output / f"generation-{child.generation:03d}.npz")
        manifest["iterations"].append(
            {
                "parent_policy_hash": parent.policy_hash,
                "policy_hash": child.policy_hash,
                "generation": child.generation,
                "rollout_report_hashes": hashes,
                "players": rows,
            }
        )
        parent = child
        (output / "training-progress.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(
            json.dumps(
                {
                    "iteration": iteration,
                    "policy_hash": parent.policy_hash,
                    "updated": [r["agent_id"] for r in rows if r["updated"]],
                }
            ),
            flush=True,
        )
    jobs = [
        MatchCollection(
            assets,
            output / f"eval-{label}-{name}",
            output / f"generation-{policy.generation:03d}.npz",
            0,
            scope,
            False,
            y,
            outlet_stance_depth_m,
            strict_handoff,
            finisher_option_learning,
            prospective_motor=prospective_motor,
            task_context_bound=task_context_bound,
            motor_lateral_limit_m=motor_lateral_limit_m,
            prospective_strike_approach=prospective_strike_approach,
            teammate_approach_clearance_m=teammate_approach_clearance_m,
            ball_x_m=x,
            keeper_distribution_preview=keeper_distribution_preview,
            per_player_options=per_player_options,
            continuous_motor_rearm=continuous_motor_rearm,
            receiver_commitment_priority=receiver_commitment_priority,
        )
        for label, policy in (("parent", initial), ("candidate", parent))
        for name, x, y in exams
    ]
    with ProcessPoolExecutor(
        max_workers=workers, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        paths = list(pool.map(collect, jobs))
    for path in paths:
        report = validate_continuous_competitive_match_growth(Path(path))
        manifest["evaluation"].append(
            {
                "course": Path(path).parent.name,
                "report_hash": report["report_hash"],
                "assessment": report["primary_assessment"],
            }
        )
    manifest["manifest_hash"] = hash_json(manifest)
    (output / "training.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--trainable-agent", action="append", required=True)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--outlet-stance-depth", type=float)
    parser.add_argument("--strict-handoff", action="store_true")
    parser.add_argument("--varied-ball-positions", action="store_true")
    parser.add_argument("--finisher-option-learning", action="store_true")
    parser.add_argument("--prospective-motor", action="store_true")
    parser.add_argument("--bound-motor-context", action="store_true")
    parser.add_argument("--balanced-motor-kickoffs", action="store_true")
    parser.add_argument("--motor-lateral-limit", type=float, default=0.60)
    parser.add_argument("--prospective-strike-approach", action="store_true")
    parser.add_argument("--teammate-approach-clearance", type=float, default=0.0)
    parser.add_argument("--buildup-kickoffs", action="store_true")
    parser.add_argument("--keeper-distribution-preview", action="store_true")
    parser.add_argument("--per-player-options", action="store_true")
    parser.add_argument("--continuous-motor-rearm", action="store_true")
    parser.add_argument("--receiver-commitment-priority", action="store_true")
    parser.add_argument(
        "--reward-shaping",
        choices=("contact_safety_v1", "motor_task_contact_v1"),
        default="contact_safety_v1",
    )
    args = parser.parse_args()
    train(
        assets=args.asset_root,
        output=args.output,
        checkpoint=args.checkpoint,
        scope=tuple(sorted(args.trainable_agent)),
        iterations=args.iterations,
        workers=args.workers,
        outlet_stance_depth_m=args.outlet_stance_depth,
        strict_handoff=args.strict_handoff,
        varied_ball_positions=args.varied_ball_positions,
        finisher_option_learning=args.finisher_option_learning,
        reward_shaping=args.reward_shaping,
        prospective_motor=args.prospective_motor,
        task_context_bound=args.bound_motor_context,
        balanced_motor_kickoffs=args.balanced_motor_kickoffs,
        motor_lateral_limit_m=args.motor_lateral_limit,
        prospective_strike_approach=args.prospective_strike_approach,
        teammate_approach_clearance_m=args.teammate_approach_clearance,
        buildup_kickoffs=args.buildup_kickoffs,
        keeper_distribution_preview=args.keeper_distribution_preview,
        per_player_options=args.per_player_options,
        continuous_motor_rearm=args.continuous_motor_rearm,
        receiver_commitment_priority=args.receiver_commitment_priority,
    )


if __name__ == "__main__":
    main()
