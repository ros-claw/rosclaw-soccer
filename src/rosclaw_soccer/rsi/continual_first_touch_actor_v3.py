"""SIM_ONLY cross-course continual first-touch actor with retention anchors.

This module migrates the seeded v2 proof into a distribution-bound actor. New
training seeds are one-use learning batches; prior seeds form a bounded anchor
set for stability. It cannot authorize real motion or policy promotion.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.rsi import online_first_touch_actor_critic_v2 as v2
from rosclaw_soccer.rsi.first_touch_candidate import (
    MAX_RESIDUAL_RAD,
    candidate_manifest,
    load_first_touch_candidate,
)
from rosclaw_soccer.rsi.first_touch_course_catalog import sample_training_courses
from rosclaw_soccer.rsi.vector_first_touch_evidence import audit_first_touch_candidate_execution
from rosclaw_soccer.sim.contracts import hash_json

SCHEMA = "rsi_isaac_continual_first_touch_actor_v3"
DOMAIN_HASH = hash_json(
    {
        "schema": "rsi_isaac_first_touch_train_domain_v1",
        "x_m": "[2.2,2.4] U [2.6,2.8]",
        "y_m": "[-0.16,0.16]",
        "abs_vx_m_s": "[0.25,0.7]",
        "courses_per_seed": 16,
        "frozen_model": "SONIC low_latency legacy-layout batched Torch",
        "action": "six joint-target residuals +/-0.08 rad before first contact",
        "activation_ceiling": "SIM_ONLY",
    }
)
MAX_PER_GENERATION_SHIFT_RAD = 0.01
MAX_GLOBAL_ANCHOR_SHIFT_RAD = 0.04
MAX_ANCHOR_SEEDS = 64
STATE_FIELDS = frozenset(
    {
        "schema",
        "activation_ceiling",
        "promotion_authorized",
        "domain_hash",
        "asset_hash",
        "sonic_qualification_hash",
        "generation",
        "actor_weights",
        "critic_weights",
        "anchor_actor_weights",
        "consumed_training_seeds",
        "consumed_audit_hashes",
        "state_hash",
    }
)


def _state(
    domain_parent: dict[str, Any],
    generation: int,
    actor: np.ndarray,
    critic: np.ndarray,
    anchor_actor: np.ndarray,
    seeds: list[int],
    audits: list[str],
) -> dict[str, Any]:
    if (
        type(generation) is not int
        or generation < 0
        or actor.shape != (4, 6)
        or critic.shape != (4,)
        or anchor_actor.shape != (4, 6)
        or not all(np.isfinite(array).all() for array in (actor, critic, anchor_actor))
        or max(float(np.max(np.abs(actor))), float(np.max(np.abs(anchor_actor)))) > 0.1
        or float(np.max(np.abs(critic))) > 10
        or type(seeds) is not list
        or not 1 <= len(seeds) <= MAX_ANCHOR_SEEDS
        or any(type(seed) is not int or not 0 <= seed < 2**32 for seed in seeds)
        or len(set(seeds)) != len(seeds)
        or type(audits) is not list
        or len(set(audits)) != len(audits)
        or any(
            not isinstance(value, str) or len(value) != 71 or not value.startswith("sha256:")
            for value in audits
        )
    ):
        raise ValueError("invalid continual actor state or retention history")
    body: dict[str, Any] = {
        "schema": SCHEMA,
        "activation_ceiling": "SIM_ONLY",
        "promotion_authorized": False,
        "domain_hash": DOMAIN_HASH,
        "asset_hash": domain_parent["asset_hash"],
        "sonic_qualification_hash": domain_parent["sonic_qualification_hash"],
        "generation": generation,
        "actor_weights": actor.tolist(),
        "critic_weights": critic.tolist(),
        "anchor_actor_weights": anchor_actor.tolist(),
        "consumed_training_seeds": seeds,
        "consumed_audit_hashes": audits,
    }
    body["state_hash"] = hash_json(body)
    return body


def load_state(data: dict[str, Any]) -> dict[str, Any]:
    committed = {key: value for key, value in data.items() if key != "state_hash"}
    if (
        set(data) != STATE_FIELDS
        or data.get("schema") != SCHEMA
        or data.get("activation_ceiling") != "SIM_ONLY"
        or data.get("promotion_authorized") is not False
        or data.get("domain_hash") != DOMAIN_HASH
        or data.get("state_hash") != hash_json(committed)
    ):
        raise ValueError("continual actor state commitment invalid")
    canonical = _state(
        data,
        data["generation"],
        np.asarray(data["actor_weights"], dtype=np.float64),
        np.asarray(data["critic_weights"], dtype=np.float64),
        np.asarray(data["anchor_actor_weights"], dtype=np.float64),
        data["consumed_training_seeds"],
        data["consumed_audit_hashes"],
    )
    if canonical["state_hash"] != data["state_hash"]:
        raise ValueError("continual actor canonical form changed")
    return canonical


def migrate_v2(parent_folder: Path, v2_data: dict[str, Any]) -> dict[str, Any]:
    parent, _ = v2._parent(parent_folder)
    verified = v2.load_state_from_dict(parent, v2_data)
    if not parent.get("torch_batch_plan_only", False) or verified["generation"] < 1:
        raise ValueError("migration requires trained batch-plan-only v2 state")
    actor = np.asarray(verified["actor_weights"], dtype=np.float64)
    return _state(
        parent,
        verified["generation"],
        actor,
        np.asarray(verified["critic_weights"], dtype=np.float64),
        actor.copy(),
        [parent["training_course_seed"]],
        verified["consumed_audit_hashes"],
    )


def _new_parent(
    folder: Path, state: dict[str, Any]
) -> tuple[dict[str, Any], tuple[tuple[float, float, float], ...]]:
    parent, courses = v2._parent(folder)
    if (
        parent.get("asset_hash") != state["asset_hash"]
        or parent.get("sonic_qualification_hash") != state["sonic_qualification_hash"]
        or parent.get("torch_batch_plan_only") is not True
        or parent["training_course_seed"] in state["consumed_training_seeds"]
    ):
        raise ValueError("new Parent violates frozen foundation or distinct-course contract")
    return parent, courses


def sample_candidate(parent_folder: Path, state: dict[str, Any], *, seed: int) -> dict[str, Any]:
    state = load_state(state)
    parent, courses = _new_parent(parent_folder, state)
    x = v2._features(courses)
    mean = v2._mean(x, np.asarray(state["actor_weights"], dtype=np.float64))
    actions = np.clip(
        mean + np.random.default_rng(seed).normal(0.0, v2.SIGMA_RAD, mean.shape),
        -MAX_RESIDUAL_RAD,
        MAX_RESIDUAL_RAD,
    )
    manifest = candidate_manifest(
        courses=courses,
        parent_report_hash=parent["report_hash"],
        actions_rad=tuple(tuple(float(value) for value in row) for row in actions),
        seed=seed,
    )
    manifest["actor_state_hash"] = state["state_hash"]
    manifest.pop("candidate_hash")
    manifest["candidate_hash"] = hash_json(manifest)
    return manifest


def deterministic_mean_candidate(parent_folder: Path, state: dict[str, Any]) -> dict[str, Any]:
    """Probe transfer to a new train seed without exploration noise."""
    state = load_state(state)
    parent, courses = _new_parent(parent_folder, state)
    mean = v2._mean(v2._features(courses), np.asarray(state["actor_weights"], dtype=np.float64))
    manifest = candidate_manifest(
        courses=courses,
        parent_report_hash=parent["report_hash"],
        actions_rad=tuple(tuple(float(value) for value in row) for row in mean),
        seed=0,
    )
    manifest["actor_state_hash"] = state["state_hash"]
    manifest["evaluation_mode"] = "FROZEN_TRANSFER_MEAN"
    manifest.pop("candidate_hash")
    manifest["candidate_hash"] = hash_json(manifest)
    return manifest


def update(
    parent_folder: Path,
    state_data: dict[str, Any],
    candidate_folder: Path,
    manifest_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    state = load_state(state_data)
    parent, courses = _new_parent(parent_folder, state)
    audit = audit_first_touch_candidate_execution(
        candidate_folder, parent_folder=parent_folder, candidate_path=manifest_path
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    candidate = load_first_touch_candidate(
        manifest_path, expected_courses=courses, parent_report_hash=parent["report_hash"]
    )
    if (
        manifest.get("actor_state_hash") != state["state_hash"]
        or audit["report_hash"] in state["consumed_audit_hashes"]
        or len(state["consumed_training_seeds"]) >= MAX_ANCHOR_SEEDS
    ):
        raise ValueError("off-policy, reused, or over-capacity physical batch")
    x = v2._features(courses)
    actor = np.asarray(state["actor_weights"], dtype=np.float64)
    critic = np.asarray(state["critic_weights"], dtype=np.float64)
    anchor_actor = np.asarray(state["anchor_actor_weights"], dtype=np.float64)
    mean = v2._mean(x, actor)
    actions = np.asarray(candidate.actions_rad)
    expected = np.clip(
        mean + np.random.default_rng(manifest["seed"]).normal(0.0, v2.SIGMA_RAD, mean.shape),
        -MAX_RESIDUAL_RAD,
        MAX_RESIDUAL_RAD,
    )
    if not np.array_equal(actions, expected):
        raise ValueError("actions differ from committed continual actor sample")
    parent_reward = np.asarray([v2._parent_reward(row) for row in parent["environments"]])
    delta = np.asarray(audit["reward_per_course"]) - parent_reward
    advantage = delta - x @ critic
    score = (actions - mean) / v2.SIGMA_RAD**2
    derivative = 1.0 - (mean / v2.MEAN_LIMIT_RAD) ** 2
    gradient = x.T @ (advantage[:, None] * score * derivative) / len(courses) - 0.05 * actor
    gradient_norm = float(np.linalg.norm(gradient))
    if gradient_norm > 5:
        gradient *= 5 / gradient_norm
    unconstrained = np.clip(actor + v2.ACTOR_STEP * gradient, -0.08, 0.08)
    anchor_x = np.concatenate(
        [
            v2._features(sample_training_courses(seed))
            for seed in [*state["consumed_training_seeds"], parent["training_course_seed"]]
        ]
    )
    old_anchor_mean = v2._mean(anchor_x, actor)
    original_anchor_mean = v2._mean(anchor_x, anchor_actor)
    direction = unconstrained - actor
    lower, upper = 0.0, 1.0
    for _ in range(32):
        middle = (lower + upper) / 2
        trial = v2._mean(anchor_x, actor + middle * direction)
        if (
            float(np.max(np.abs(trial - old_anchor_mean))) <= MAX_PER_GENERATION_SHIFT_RAD
            and float(np.max(np.abs(trial - original_anchor_mean))) <= MAX_GLOBAL_ANCHOR_SHIFT_RAD
        ):
            lower = middle
        else:
            upper = middle
    next_actor = actor + lower * direction
    next_mean = v2._mean(anchor_x, next_actor)
    per_generation_shift = float(np.max(np.abs(next_mean - old_anchor_mean)))
    global_shift = float(np.max(np.abs(next_mean - original_anchor_mean)))
    if (
        per_generation_shift > MAX_PER_GENERATION_SHIFT_RAD + 1e-9
        or global_shift > MAX_GLOBAL_ANCHOR_SHIFT_RAD + 1e-9
    ):
        raise ValueError("stability-plasticity retention contract failed")
    next_state = _state(
        parent,
        state["generation"] + 1,
        next_actor,
        np.clip(critic + v2.CRITIC_STEP * (x.T @ advantage / len(courses)), -5, 5),
        anchor_actor,
        [*state["consumed_training_seeds"], parent["training_course_seed"]],
        [*state["consumed_audit_hashes"], audit["report_hash"]],
    )
    report: dict[str, Any] = {
        "schema": "rsi_isaac_continual_first_touch_update_v3",
        "activation_ceiling": "SIM_ONLY",
        "promotion_authorized": False,
        "before_state_hash": state["state_hash"],
        "after_state_hash": next_state["state_hash"],
        "new_training_seed": parent["training_course_seed"],
        "parent_report_hash": parent["report_hash"],
        "candidate_audit_hash": audit["report_hash"],
        "physical_episode_count": 16,
        "parent_clean_count": int(np.sum(parent_reward == 1.0)),
        "candidate_clean_count": audit["candidate_clean_foot_only_count"],
        "mean_reward_delta": float(np.mean(delta)),
        "actor_gradient_norm_before_clip": gradient_norm,
        "maximum_per_generation_anchor_shift_rad": per_generation_shift,
        "maximum_global_anchor_shift_rad": global_shift,
        "fresh_opened": False,
    }
    report["report_hash"] = hash_json(report)
    return next_state, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-folder", required=True, type=Path)
    parser.add_argument("--state", type=Path)
    parser.add_argument("--migrate-v2", type=Path)
    parser.add_argument("--candidate-folder", type=Path)
    parser.add_argument("--candidate-manifest", type=Path)
    parser.add_argument("--sample-seed", type=int)
    parser.add_argument("--deterministic-mean", action="store_true")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--output-report", type=Path)
    args = parser.parse_args()
    if args.migrate_v2 is not None:
        if (
            args.state is not None
            or args.candidate_folder is not None
            or args.sample_seed is not None
            or args.deterministic_mean
        ):
            parser.error("migration must be isolated")
        result = migrate_v2(
            args.parent_folder, json.loads(args.migrate_v2.read_text(encoding="utf-8"))
        )
        report = None
    else:
        if args.state is None:
            parser.error("sampling/updating requires a committed continual state")
        state = load_state(json.loads(args.state.read_text(encoding="utf-8")))
        if args.sample_seed is not None or args.deterministic_mean:
            if args.candidate_folder is not None or args.candidate_manifest is not None:
                parser.error("sampling cannot consume execution evidence")
            if args.sample_seed is not None and args.deterministic_mean:
                parser.error("choose stochastic sample or deterministic mean")
            result = (
                deterministic_mean_candidate(args.parent_folder, state)
                if args.deterministic_mean
                else sample_candidate(args.parent_folder, state, seed=args.sample_seed)
            )
            report = None
        else:
            if args.candidate_folder is None or args.candidate_manifest is None:
                parser.error("update requires physical execution and manifest")
            result, report = update(
                args.parent_folder, state, args.candidate_folder, args.candidate_manifest
            )
    if report is not None and args.output_report is None:
        parser.error("update requires report output")
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
    if report is not None:
        assert args.output_report is not None
        with args.output_report.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, sort_keys=True)
            stream.write("\n")
    print(json.dumps({"state_hash": result.get("state_hash"), "report": report}, sort_keys=True))


if __name__ == "__main__":
    main()
