"""Rebuild each scoped continuous-match PPO update; no promotion authority."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.growth.near_ball_residual import NearBallResidualPolicy
from rosclaw_soccer.sim.contracts import hash_json
from rosclaw_soccer.training.continuous_competitive_match_growth import (
    validate_continuous_competitive_match_growth,
)
from rosclaw_soccer.training.continuous_match_residual_ppo import (
    collection_options,
    collection_world,
    evaluation_kickoffs,
    training_kickoffs,
)
from rosclaw_soccer.training.near_ball_plasticity import recorded_training_scope
from rosclaw_soccer.training.near_ball_residual_ppo import update_private_actors


def audit(root: Path) -> dict[str, Any]:
    manifest = json.loads((root / "training.json").read_text())
    commitment = manifest.pop("manifest_hash")
    if (
        commitment != hash_json(manifest)
        or manifest["schema"] != "rosclaw_soccer.continuous_match_residual_ppo.v1"
        or manifest["activation_ceiling"] != "SIM_ONLY"
        or manifest["promotion_eligible"] is not False
    ):
        raise ValueError("continuous learning manifest commitment differs")
    iterations = manifest["iterations"]
    if not 1 <= len(iterations) <= 20:
        raise ValueError("invalid training iteration count")
    initial_generation = iterations[0]["generation"] - 1
    parent = NearBallResidualPolicy.load(root / f"generation-{initial_generation:03d}.npz")
    scope = recorded_training_scope(manifest["trainable_agent_ids"], parent.agent_ids)
    if (
        scope is None
        or list(scope) != manifest["exploration_agent_ids"]
        or parent.policy_hash != manifest["initial_policy_hash"]
        or manifest["optimizer_epochs"] != 8
        or manifest["gamma"] != 0.997
        or manifest["trace_decay"] != 0.997
        or manifest["reward_shaping"] not in ("contact_safety_v1", "motor_task_contact_v1")
    ):
        raise ValueError("continuous learning scope or protocol differs")
    checked = []
    coverage = {
        agent: {"active_samples": 0, "learning_samples": 0, "optimizer_updates": 0}
        for agent in parent.agent_ids
    }
    offsets = manifest["training_ball_y_m"]
    kickoffs = training_kickoffs(
        varied=manifest.get("varied_ball_positions", offsets == [0.0, -0.02, 0.02, -0.04, 0.04]),
        balanced_motor=manifest.get("balanced_motor_kickoffs", False),
        buildup=manifest.get("buildup_kickoffs", False),
    )
    if offsets != [y for _, y in kickoffs] or manifest.get(
        "training_ball_xy_m", [list(xy) for xy in kickoffs]
    ) != [list(xy) for xy in kickoffs]:
        raise ValueError("unknown committed ball-position curriculum")
    world = collection_world(
        manifest.get("outlet_stance_depth_m"),
        manifest.get("strict_handoff", False),
        manifest.get("finisher_option_learning", False),
        manifest.get("prospective_strike_approach", False),
        manifest.get("teammate_approach_clearance_m", 0.0),
    )
    world_hash = world.config_hash
    option_hash = collection_options(
        world,
        prospective=manifest.get("prospective_motor", False),
        bound_context=manifest.get("task_context_bound", False),
        lateral_limit_m=manifest.get("motor_lateral_limit_m", 0.6),
        per_player_options=manifest.get("per_player_options", False),
        continuous_motor_rearm=manifest.get("continuous_motor_rearm", False),
    ).config_hash
    bound_option_hash = manifest.get("collection_option_config_hash")
    if bound_option_hash is not None and bound_option_hash != option_hash:
        raise ValueError("committed motor option differs")
    if (
        any(
            manifest.get(k, False)
            for k in (
                "prospective_motor",
                "task_context_bound",
                "balanced_motor_kickoffs",
                "per_player_options",
                "continuous_motor_rearm",
            )
        )
        and bound_option_hash is None
    ):
        raise ValueError("new motor curriculum requires an option commitment")
    if manifest.get("collection_world_config_hash", world_hash) != world_hash:
        raise ValueError("committed physical world differs")
    preview = manifest.get("keeper_distribution_preview", False)
    fixture_hash = manifest.get("collection_fixture_hash")
    if type(preview) is not bool or (
        (preview or manifest.get("buildup_kickoffs", False))
        and (
            not isinstance(fixture_hash, str)
            or len(fixture_hash) != 71
            or not fixture_hash.startswith("sha256:")
        )
    ):
        raise ValueError("buildup learning requires an explicit fixture commitment")
    if fixture_hash is not None and not {
        "evaluation_case_names",
        "evaluation_ball_xy_m",
        "evaluation_ball_y_m",
        "evaluation",
    }.issubset(manifest):
        raise ValueError("fixture-bound learning requires held-out evaluation commitments")

    def fixture_matches(report: dict[str, Any]) -> bool:
        return bool(
            (fixture_hash is None or report["fixture_hash"] == fixture_hash)
            and all(
                cell["tactical_profile"].get("keeper_distribution_preview", False) is preview
                for cell in report["agent_cells"]
            )
        )

    for iteration, row in enumerate(iterations):
        if (
            row["generation"] != parent.generation + 1
            or row["parent_policy_hash"] != parent.policy_hash
        ):
            raise ValueError("continuous policy lineage differs")
        traces, proofs = [], []
        for index in range(4):
            directory = root / f"train-{iteration:03d}-{index}"
            report = validate_continuous_competitive_match_growth(
                directory / "continuous-match-exam.json"
            )
            seed = 196000 + 4 * iteration + index
            if (
                report["near_ball_policy_hash"] != parent.policy_hash
                or report["sampling"]
                != {"explore": True, "seed": seed, "exploration_agent_ids": list(scope)}
                or report["passed"] is not False
                or not report["exact_replay"]
                or report["world_config_hash"] != world_hash
                or not fixture_matches(report)
                or (
                    bound_option_hash is not None
                    and report["option_config_hash"] != bound_option_hash
                )
                or report["scenario"]["ball_initial_position_m"][:2]
                != list(kickoffs[(4 * iteration + index) % len(kickoffs)])
            ):
                raise ValueError("continuous collection binding differs")
            with np.load(directory / "primary.npz", allow_pickle=False) as archive:
                trace = {k: archive[k] for k in archive.files}
            mask = np.asarray([agent in scope for agent in parent.agent_ids], dtype=bool)
            recorded = trace["residual_exploration_mask"]
            if recorded.dtype != np.bool_ or not np.array_equal(
                recorded, np.broadcast_to(mask, (len(trace["time"]), len(mask)))
            ):
                raise ValueError("physical exploration scope differs")
            rng = np.random.default_rng(seed)
            for frame, obs in enumerate(trace["residual_observations"]):
                values = parent.act(obs, rng, explore=True, exploration_mask=mask)
                for expected, key in zip(
                    values,
                    ("residual_latent", "residual_log_probability", "residual_value"),
                    strict=True,
                ):
                    if not np.array_equal(expected, trace[key][frame]):
                        raise ValueError("seeded scoped collection cannot be reproduced")
            traces.append(trace)
            proofs.append(report["report_hash"])
        if proofs != row["rollout_report_hashes"]:
            raise ValueError("optimizer dataset ordering differs")
        rebuilt, players = update_private_actors(
            parent,
            traces,
            epochs=8,
            gamma=0.997,
            trace_decay=0.997,
            reward_shaping=manifest["reward_shaping"],
            trainable_agent_ids=scope,
        )
        saved = NearBallResidualPolicy.load(root / f"generation-{rebuilt.generation:03d}.npz")
        if (
            rebuilt.policy_hash != saved.policy_hash
            or saved.policy_hash != row["policy_hash"]
            or players != row["players"]
        ):
            raise ValueError("recorded optimizer and Core lease proofs cannot be reproduced")
        for player in players:
            counts = coverage[player["agent_id"]]
            counts["active_samples"] += player["active_samples"]
            counts["learning_samples"] += player.get("learning_samples", player["active_samples"])
            counts["optimizer_updates"] += int("core_plasticity" in player)
        checked.append(saved.policy_hash)
        parent = saved
    if "evaluation_case_names" in manifest:
        exams = evaluation_kickoffs(buildup=manifest.get("buildup_kickoffs", False))
        if (
            manifest["evaluation_case_names"] != [name for name, _, _ in exams]
            or manifest["evaluation_ball_xy_m"] != [[x, y] for _, x, y in exams]
            or manifest["evaluation_ball_y_m"] != [y for _, _, y in exams]
            or len(manifest["evaluation"]) != 2 * len(exams)
        ):
            raise ValueError("held-out curriculum differs")
        entries = iter(manifest["evaluation"])
        for label, policy_hash in (
            ("parent", manifest["initial_policy_hash"]),
            ("candidate", parent.policy_hash),
        ):
            for name, x, y in exams:
                row = next(entries)
                course = f"eval-{label}-{name}"
                report = validate_continuous_competitive_match_growth(
                    root / course / "continuous-match-exam.json"
                )
                if (
                    row["course"] != course
                    or row["report_hash"] != report["report_hash"]
                    or row["assessment"] != report["primary_assessment"]
                    or report["sampling"]
                    != {"explore": False, "seed": 0, "exploration_agent_ids": None}
                    or report["near_ball_policy_hash"] != policy_hash
                    or report["world_config_hash"] != world_hash
                    or report["option_config_hash"] != option_hash
                    or not fixture_matches(report)
                    or not report["exact_replay"]
                    or report["scenario"]["ball_initial_position_m"][:2] != [x, y]
                ):
                    raise ValueError("held-out physical evaluation binding differs")
    result = {
        "training_manifest_hash": commitment,
        "exact_rebuilt_policy_hashes": checked,
        "verified_collection_pairs": 4 * len(checked),
        "verified_core_updates": sum(c["optimizer_updates"] for c in coverage.values()),
        "role_learning_coverage": coverage,
        "requested_but_never_updated": [
            agent for agent in scope if coverage[agent]["optimizer_updates"] == 0
        ],
        "activation_ceiling": "SIM_ONLY",
        "promotion_eligible": False,
    }
    result["report_hash"] = hash_json(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.root), indent=2))


if __name__ == "__main__":
    main()
