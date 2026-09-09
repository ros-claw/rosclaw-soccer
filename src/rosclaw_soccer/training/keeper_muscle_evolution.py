"""SIM_ONLY reward-driven last-layer search, not actor-critic or promotion.

Every candidate is tested in physical MuJoCo on the difficult training shot
and an acquired-skill retention shot. No launcher clock is a policy input.
Full trajectories and rejected candidates are preserved outside the repository.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.providers.g1.keeper_muscle_actor import KeeperMuscleActor
from rosclaw_soccer.providers.g1.shared_keeper_reach import SharedKeeperReachConfig
from rosclaw_soccer.sim.contracts import hash_bytes
from rosclaw_soccer.training.shared_keeper_reach_exam import run_case
from rosclaw_soccer.world.goalkeeper_glove_material import GoalkeeperGloveMaterial


def retention_null_direction(
    actor: KeeperMuscleActor, anchors: np.ndarray, novel: np.ndarray
) -> tuple[np.ndarray, dict[str, Any]]:
    """Preserve finite rehearsal outputs, NOT all future closed-loop states."""
    features = []
    for observations in (anchors, novel):
        observations = np.asarray(observations, dtype=np.float64)
        if (
            observations.ndim != 2
            or observations.shape[1] != 65
            or len(observations) == 0
            or not np.isfinite(observations).all()
            or np.max(abs(observations)) > 5
        ):
            raise ValueError("invalid finite rehearsal observations")
        x = observations.copy()
        for w, b in actor.layers[:2]:
            x = np.tanh(x @ w.T + b)
        features.append(np.column_stack((x, np.ones(len(x)))))
    old, new = features
    _, singular, vectors = np.linalg.svd(old, full_matrices=True)
    rank = int(np.sum(singular > 1e-10))
    null = vectors[rank:].T
    direction = null @ (null.T @ np.mean(new, axis=0))
    norm = float(np.linalg.norm(direction))
    if norm < 1e-8:
        raise ValueError("no distinct plasticity direction remains in rehearsal nullspace")
    direction /= norm
    error = float(np.max(abs(old @ direction)))
    if error > 1e-8:
        raise ValueError("rehearsal projection failed numerical preservation check")
    return direction, dict(
        anchor_count=len(old),
        anchor_rank=rank,
        remaining_dimension=65 - rank,
        anchor_projection_max=error,
        novel_mean_response=float(np.mean(new @ direction)),
    )


def physical_reward(result: dict[str, Any], *, signed_clearance_feedback: bool = False) -> float:
    """Dense reach signal cannot compensate for a fall or replace save gates."""
    keys = (
        "minimum_pelvis_m",
        "peak_tilt_rad",
        "closest_incoming_glove_surface_m",
        "outward_speed_mps",
    )
    if not all(np.isfinite(result[k]) for k in keys):
        raise ValueError("nonfinite physical feedback")
    if (
        result.get("physical_safe") is not True
        or result["minimum_pelvis_m"] <= 0.55
        or result["peak_tilt_rad"] >= 0.8
    ):
        return -10.0
    distance = float(np.clip(result["closest_incoming_glove_surface_m"], 0, 2))
    clearance_reward = 0.0
    if signed_clearance_feedback and result["first_robot_contact_glove"] is True:
        signed = result.get("signed_early_clearance_speed_mps")
        if signed is not None and not np.isfinite(signed):
            raise ValueError("nonfinite signed physical clearance feedback")
        clearance_reward = 0.5 * float(np.clip(-6 if signed is None else signed, -6, 3))
    return float(
        clearance_reward
        - 3 * distance
        + 1.0 * (result["first_robot_contact_glove"] is True)
        + 3 * bool(result["completed_hand_save"])
        + 2 * bool(result["stable_save"])
        + min(float(result["outward_speed_mps"]), 3) * bool(result["completed_hand_save"])
        - bool(result["goal_crossed"])
    )


def train(
    root: Path,
    parent: Path,
    config: SharedKeeperReachConfig,
    output: Path,
    *,
    generations: int = 3,
    population: int = 6,
    seed: int = 230,
    protect_rehearsal: bool = False,
    initial_sigma: float = 0.10,
    sigma_floor: float = 0.025,
    signed_clearance_feedback: bool = False,
    bootstrap: bool = False,
) -> dict[str, Any]:
    if not 1 <= generations <= 20 or not 4 <= population <= 32:
        raise ValueError("invalid bounded simulation search budget")
    if bootstrap and protect_rehearsal:
        raise ValueError("bootstrap cannot claim protection of a qualified acquired skill")
    if not (
        np.isfinite((initial_sigma, sigma_floor)).all()
        and 0.0005 <= sigma_floor <= initial_sigma <= 0.2
    ):
        raise ValueError("invalid bounded exploration scale")
    actor = KeeperMuscleActor(parent)
    output.mkdir(parents=True, exist_ok=False)
    payload = actor.metadata
    original = np.asarray(payload["layers"][-1]["bias"], dtype=float)
    rng = np.random.default_rng(seed)
    mean, sigma = np.zeros(14), np.full(14, initial_sigma)
    incumbent = np.zeros(14)
    best_score = -float("inf")
    records = []
    material = GoalkeeperGloveMaterial(0.0075, 0.15, 0.08)
    direction = None
    projection = None
    if protect_rehearsal:
        traces = []
        for height in (1.3, 1.5):
            directory = output / f"rehearsal-{height}"
            result = run_case(
                root,
                directory,
                team="red",
                lateral=0,
                height=height,
                enabled=True,
                config=replace(config, muscle_actor_path=str(parent)),
                glove_material=material,
            )
            if height == 1.3 and not result["stable_save"]:
                raise ValueError("parent failed the skill claimed for retention")
            if result["muscle_policy_hash"] != actor.policy_hash:
                raise ValueError("rehearsal parent changed after its identity was bound")
            with np.load(directory / "trajectory.npz", allow_pickle=False) as trace:
                traces.append(trace["muscle_observations"])
        direction, projection = retention_null_direction(actor, *traces)
        scale = min(20.0, 1 / max(projection["novel_mean_response"], 0.001))
        direction *= scale
        projection["direction_scale"] = scale
    for generation in range(generations):
        proposals = [
            incumbent.copy(),
            *np.clip(rng.normal(mean, sigma, (population - 1, 14)), -0.35, 0.35),
        ]
        evaluated = []
        for index, delta in enumerate(proposals):
            candidate = json.loads(json.dumps(payload))
            candidate["layers"][-1]["bias"] = (original + delta).tolist()
            if direction is not None:
                update = delta[:, None] * direction
                candidate["layers"][-1]["bias"] = (original + update[:, 64]).tolist()
                candidate["layers"][-1]["weight"] = (
                    np.asarray(payload["layers"][-1]["weight"]) + update[:, :64]
                ).tolist()
            candidate.update(
                parent_policy_hash=actor.policy_hash,
                method="physical_reward_cross_entropy_last_layer_search",
                generation=generation,
                candidate=index,
                heldout_physics_passed=False,
                promotion_authorized=False,
                rehearsal_projection=projection,
            )
            directory = output / f"g{generation:02d}-c{index:02d}"
            directory.mkdir()
            path = directory / "actor.json"
            path.write_text(json.dumps(candidate, separators=(",", ":")) + "\n")
            loaded = KeeperMuscleActor(path)
            results = []
            for height in (1.3, 1.5):
                result = run_case(
                    root,
                    directory / f"height-{height}",
                    team="red",
                    lateral=0,
                    height=height,
                    enabled=True,
                    config=replace(config, muscle_actor_path=str(path)),
                    glove_material=material,
                )
                if result["muscle_policy_hash"] != loaded.policy_hash:
                    raise RuntimeError("executed candidate identity mismatch")
                results.append(result)
            # Acquired skill is a hard retention constraint, not an average
            # reward that can hide catastrophic forgetting on the easier shot.
            retained = bool(results[0]["stable_save"])
            score = (
                physical_reward(results[1], signed_clearance_feedback=signed_clearance_feedback)
                if retained
                else -20.0
            )
            safe_bootstrap = bootstrap and all(r["physical_safe"] is True for r in results)
            if safe_bootstrap:
                score = float(
                    np.mean(
                        [
                            physical_reward(r, signed_clearance_feedback=signed_clearance_feedback)
                            for r in results
                        ]
                    )
                )
            record = dict(
                generation=generation,
                candidate=index,
                retained=retained,
                bootstrap=bootstrap,
                score=score,
                delta=delta.tolist(),
                policy_hash=loaded.policy_hash,
                training_results=[
                    {
                        k: r[k]
                        for k in (
                            "height",
                            "completed_hand_save",
                            "stable_save",
                            "outward_speed_mps",
                            "closest_incoming_glove_surface_m",
                            "trajectory_hash",
                            "signed_early_clearance_speed_mps",
                        )
                    }
                    for r in results
                ],
            )
            records.append(record)
            (directory / "feedback.json").write_text(json.dumps(record, indent=2) + "\n")
            print(json.dumps(record), flush=True)
            if retained or safe_bootstrap:
                evaluated.append((score, delta, all(r["completed_hand_save"] for r in results)))
            if score > best_score:
                best_score, incumbent = score, delta.copy()
        if not evaluated:
            raise ValueError("no candidate passed the declared safety/retention requirement")
        elite = sorted(evaluated, key=lambda item: item[0], reverse=True)[:2]
        # Once a completed save exists, do not average its parameters with
        # missed shots. Keep exploring locally around successful candidates.
        successful = [item for item in evaluated if item[2]]
        if successful and not bootstrap:
            elite = sorted(successful, key=lambda item: item[0], reverse=True)[:2]
        mean = np.mean([item[1] for item in elite], axis=0)
        sigma = np.maximum(np.std([item[1] for item in elite], axis=0), sigma_floor)
        report = dict(
            schema="keeper-muscle-evolution.v1",
            activation_ceiling="SIM_ONLY",
            candidate_promoted=False,
            bootstrap=bootstrap,
            qualified_retention_required=not bootstrap,
            heldout_physics_passed=False,
            parent_policy_hash=actor.policy_hash,
            seed=seed,
            completed_generations=generation + 1,
            best_training_score=best_score,
            incumbent_delta=incumbent.tolist(),
            records=records,
            source_hash=str(hash_bytes(Path(__file__).read_bytes())),
            rehearsal_projection=projection,
            initial_sigma=initial_sigma,
            sigma_floor=sigma_floor,
            reward_contract="signed_pre_obstacle_clearance.v1"
            if signed_clearance_feedback
            else "legacy.v1",
        )
        (output / "training-report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--config-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--generations", type=int, default=3)
    parser.add_argument("--population", type=int, default=6)
    parser.add_argument("--protect-rehearsal", action="store_true")
    parser.add_argument("--initial-sigma", type=float, default=0.10)
    parser.add_argument("--sigma-floor", type=float, default=0.025)
    parser.add_argument("--signed-clearance-feedback", action="store_true")
    parser.add_argument(
        "--bootstrap",
        action="store_true",
        help="Learn from unqualified warm start; no acquired-skill claim",
    )
    args = parser.parse_args()
    config = SharedKeeperReachConfig(**json.loads(args.config_result.read_text())["config"])
    train(
        args.asset_root,
        args.parent,
        config,
        args.output,
        generations=args.generations,
        population=args.population,
        protect_rehearsal=args.protect_rehearsal,
        initial_sigma=args.initial_sigma,
        sigma_floor=args.sigma_floor,
        signed_clearance_feedback=args.signed_clearance_feedback,
        bootstrap=args.bootstrap,
    )


if __name__ == "__main__":
    main()
