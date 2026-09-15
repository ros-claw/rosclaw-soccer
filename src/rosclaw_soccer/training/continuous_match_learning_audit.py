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
        or manifest["reward_shaping"] != "contact_safety_v1"
    ):
        raise ValueError("continuous learning scope or protocol differs")
    checked = []
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
            reward_shaping="contact_safety_v1",
            trainable_agent_ids=scope,
        )
        saved = NearBallResidualPolicy.load(root / f"generation-{rebuilt.generation:03d}.npz")
        if (
            rebuilt.policy_hash != saved.policy_hash
            or saved.policy_hash != row["policy_hash"]
            or players != row["players"]
        ):
            raise ValueError("recorded optimizer and Core lease proofs cannot be reproduced")
        checked.append(saved.policy_hash)
        parent = saved
    result = {
        "training_manifest_hash": commitment,
        "exact_rebuilt_policy_hashes": checked,
        "verified_collection_pairs": 4 * len(checked),
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
