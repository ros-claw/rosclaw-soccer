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
from rosclaw_soccer.rsi.vector_first_touch_evidence import (
    audit_first_touch_candidate_execution,
    audit_vector_first_touch,
)
from rosclaw_soccer.sim.contracts import hash_json

SCHEMA = "rsi_isaac_continual_first_touch_actor_v3_1"
PROVISIONAL_SCHEMA = "rsi_isaac_continual_first_touch_provisional_v1"
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
        "origin_parent_report_hash",
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
    origin_parent_hash: str,
    qualification_hash: str | None = None,
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
        or not isinstance(origin_parent_hash, str)
        or len(origin_parent_hash) != 71
        or not origin_parent_hash.startswith("sha256:")
        or (generation == 0 and qualification_hash is not None)
        or (
            qualification_hash is not None
            and (len(qualification_hash) != 71 or not qualification_hash.startswith("sha256:"))
        )
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
        "origin_parent_report_hash": origin_parent_hash,
        "generation": generation,
        "actor_weights": actor.tolist(),
        "critic_weights": critic.tolist(),
        "anchor_actor_weights": anchor_actor.tolist(),
        "consumed_training_seeds": seeds,
        "consumed_audit_hashes": audits,
    }
    if qualification_hash is not None:
        body["qualification_hash"] = qualification_hash
    body["state_hash"] = hash_json(body)
    return body


def load_state(data: dict[str, Any]) -> dict[str, Any]:
    committed = {key: value for key, value in data.items() if key != "state_hash"}
    if (
        set(data)
        != (STATE_FIELDS if data.get("generation") == 0 else STATE_FIELDS | {"qualification_hash"})
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
        data["origin_parent_report_hash"],
        data.get("qualification_hash"),
    )
    if canonical["state_hash"] != data["state_hash"]:
        raise ValueError("continual actor canonical form changed")
    return canonical


def initial_state(parent_folder: Path) -> dict[str, Any]:
    """Start from the authenticated frozen Parent, never a failed learned mean."""
    parent, _ = v2._parent(parent_folder)
    if parent.get("torch_batch_plan_only") is not True:
        raise ValueError("continual actor requires frozen batch-plan-only Parent")
    return _state(
        parent,
        0,
        np.zeros((4, 6)),
        np.zeros(4),
        np.zeros((4, 6)),
        [parent["training_course_seed"]],
        [],
        parent["report_hash"],
    )


def migrate_v2(
    parent_folder: Path,
    v2_data: dict[str, Any],
    *,
    mean_folder: Path,
    mean_manifest_path: Path,
) -> dict[str, Any]:
    parent, _ = v2._parent(parent_folder)
    verified = v2.load_state_from_dict(parent, v2_data)
    if not parent.get("torch_batch_plan_only", False) or verified["generation"] < 1:
        raise ValueError("migration requires trained batch-plan-only v2 state")
    manifest = json.loads(mean_manifest_path.read_text(encoding="utf-8"))
    expected = v2.deterministic_mean_candidate(parent_folder, verified)
    audited = audit_first_touch_candidate_execution(
        mean_folder, parent_folder=parent_folder, candidate_path=mean_manifest_path
    )
    parent_clean = audit_vector_first_touch(parent_folder)["clean_foot_only_episode_count"]
    if (
        manifest != expected
        or audited["candidate_hash"] != expected["candidate_hash"]
        or audited["candidate_clean_foot_only_count"] < parent_clean
    ):
        raise ValueError("deterministic learned mean failed physical retention gate")
    actor = np.asarray(verified["actor_weights"], dtype=np.float64)
    return _state(
        parent,
        verified["generation"],
        actor,
        np.asarray(verified["critic_weights"], dtype=np.float64),
        actor.copy(),
        [parent["training_course_seed"]],
        verified["consumed_audit_hashes"],
        parent["report_hash"],
        audited["report_hash"],
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


def _provisional_state(active_body: dict[str, Any], parent_state_hash: str) -> dict[str, Any]:
    result = {
        key: value
        for key, value in active_body.items()
        if key not in {"schema", "state_hash", "qualification_hash"}
    }
    result["schema"] = PROVISIONAL_SCHEMA
    result["status"] = "PROVISIONAL"
    result["parent_state_hash"] = parent_state_hash
    result["state_hash"] = hash_json(result)
    return result


def load_provisional(data: dict[str, Any]) -> dict[str, Any]:
    if (
        set(data) != STATE_FIELDS | {"status", "parent_state_hash"}
        or data.get("schema") != PROVISIONAL_SCHEMA
        or data.get("status") != "PROVISIONAL"
        or type(data.get("generation")) is not int
        or data["generation"] < 1
        or not isinstance(data.get("parent_state_hash"), str)
        or len(data["parent_state_hash"]) != 71
        or data.get("state_hash")
        != hash_json({key: value for key, value in data.items() if key != "state_hash"})
    ):
        raise ValueError("provisional actor commitment invalid")
    canonical = _state(
        data,
        data["generation"],
        np.asarray(data["actor_weights"], dtype=np.float64),
        np.asarray(data["critic_weights"], dtype=np.float64),
        np.asarray(data["anchor_actor_weights"], dtype=np.float64),
        data["consumed_training_seeds"],
        data["consumed_audit_hashes"],
        data["origin_parent_report_hash"],
    )
    if _provisional_state(canonical, data["parent_state_hash"])["state_hash"] != data["state_hash"]:
        raise ValueError("provisional actor canonical form changed")
    return data


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


def provisional_mean_candidate(
    parent_folder: Path, provisional_data: dict[str, Any]
) -> dict[str, Any]:
    """Return a no-noise physical probe, not an active policy sample."""
    state = load_provisional(provisional_data)
    parent, courses = v2._parent(parent_folder)
    if (
        parent.get("asset_hash") != state["asset_hash"]
        or parent.get("sonic_qualification_hash") != state["sonic_qualification_hash"]
        or parent.get("torch_batch_plan_only") is not True
        or parent["training_course_seed"] not in state["consumed_training_seeds"]
    ):
        raise ValueError("provisional mean requires a consumed, same-foundation training seed")
    mean = v2._mean(v2._features(courses), np.asarray(state["actor_weights"], dtype=np.float64))
    manifest = candidate_manifest(
        courses=courses,
        parent_report_hash=parent["report_hash"],
        actions_rad=tuple(tuple(float(value) for value in row) for row in mean),
        seed=0,
    )
    manifest["actor_state_hash"] = state["state_hash"]
    manifest["evaluation_mode"] = "PROVISIONAL_MEAN_RETENTION"
    manifest.pop("candidate_hash")
    manifest["candidate_hash"] = hash_json(manifest)
    return manifest


def qualify_provisional(
    previous_data: dict[str, Any],
    provisional_data: dict[str, Any],
    *,
    old_parent_folder: Path,
    old_mean_folder: Path,
    old_mean_manifest: Path,
    new_parent_folder: Path,
    new_mean_folder: Path,
    new_mean_manifest: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Activate only if deterministic physical means retain both old/new Parent skill."""
    previous = load_state(previous_data)
    provisional = load_provisional(provisional_data)
    old_parent, _ = v2._parent(old_parent_folder)
    new_parent, _ = v2._parent(new_parent_folder)
    if (
        provisional["parent_state_hash"] != previous["state_hash"]
        or provisional["generation"] != previous["generation"] + 1
        or provisional["origin_parent_report_hash"] != previous["origin_parent_report_hash"]
        or old_parent["report_hash"] != previous["origin_parent_report_hash"]
        or provisional["consumed_training_seeds"][:-1] != previous["consumed_training_seeds"]
        or provisional["consumed_audit_hashes"][:-1] != previous["consumed_audit_hashes"]
        or new_parent["training_course_seed"] != provisional["consumed_training_seeds"][-1]
    ):
        raise ValueError("provisional lineage or retention Parent changed")
    audit_hashes = []
    for parent_folder, mean_folder, manifest_path in (
        (old_parent_folder, old_mean_folder, old_mean_manifest),
        (new_parent_folder, new_mean_folder, new_mean_manifest),
    ):
        expected = provisional_mean_candidate(parent_folder, provisional)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        audited = audit_first_touch_candidate_execution(
            mean_folder, parent_folder=parent_folder, candidate_path=manifest_path
        )
        parent_audit = audit_vector_first_touch(parent_folder)
        mean_report = json.loads((mean_folder / "report.json").read_text(encoding="utf-8"))
        if (
            manifest != expected
            or audited["candidate_hash"] != expected["candidate_hash"]
            or audited["candidate_clean_foot_only_count"]
            < parent_audit["clean_foot_only_episode_count"]
            or any(row["minimum_pelvis_z_m"] < 0.65 for row in mean_report["environments"])
        ):
            raise ValueError("deterministic old/new physical retention gate rejected update")
        audit_hashes.append(audited["report_hash"])
    qualification = hash_json(
        {
            "schema": "rsi_isaac_continual_actor_two_seed_qualification_v1",
            "previous_state_hash": previous["state_hash"],
            "provisional_state_hash": provisional["state_hash"],
            "old_mean_audit_hash": audit_hashes[0],
            "new_mean_audit_hash": audit_hashes[1],
            "activation_ceiling": "SIM_ONLY",
        }
    )
    active = _state(
        provisional,
        provisional["generation"],
        np.asarray(provisional["actor_weights"], dtype=np.float64),
        np.asarray(provisional["critic_weights"], dtype=np.float64),
        np.asarray(provisional["anchor_actor_weights"], dtype=np.float64),
        provisional["consumed_training_seeds"],
        provisional["consumed_audit_hashes"],
        provisional["origin_parent_report_hash"],
        qualification,
    )
    report = {
        "schema": "rsi_isaac_continual_actor_qualification_v1",
        "activation_ceiling": "SIM_ONLY",
        "promotion_authorized": False,
        "previous_state_hash": previous["state_hash"],
        "provisional_state_hash": provisional["state_hash"],
        "qualified_state_hash": active["state_hash"],
        "qualification_hash": qualification,
        "old_mean_audit_hash": audit_hashes[0],
        "new_mean_audit_hash": audit_hashes[1],
        "fresh_opened": False,
    }
    report["report_hash"] = hash_json(report)
    return active, report


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
    next_state = _provisional_state(
        _state(
            parent,
            state["generation"] + 1,
            next_actor,
            np.clip(critic + v2.CRITIC_STEP * (x.T @ advantage / len(courses)), -5, 5),
            anchor_actor,
            [*state["consumed_training_seeds"], parent["training_course_seed"]],
            [*state["consumed_audit_hashes"], audit["report_hash"]],
            state["origin_parent_report_hash"],
        ),
        state["state_hash"],
    )
    report: dict[str, Any] = {
        "schema": "rsi_isaac_continual_first_touch_update_v3",
        "activation_ceiling": "SIM_ONLY",
        "promotion_authorized": False,
        "next_state_status": "PROVISIONAL_REQUIRES_TWO_SEED_MEAN_RETENTION",
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
    parser.add_argument("--initialize", action="store_true")
    parser.add_argument("--migrate-v2", type=Path)
    parser.add_argument("--retention-mean-folder", type=Path)
    parser.add_argument("--retention-mean-manifest", type=Path)
    parser.add_argument("--candidate-folder", type=Path)
    parser.add_argument("--candidate-manifest", type=Path)
    parser.add_argument("--sample-seed", type=int)
    parser.add_argument("--deterministic-mean", action="store_true")
    parser.add_argument("--provisional-mean", action="store_true")
    parser.add_argument("--qualify-previous-state", type=Path)
    parser.add_argument("--old-parent-folder", type=Path)
    parser.add_argument("--old-mean-folder", type=Path)
    parser.add_argument("--old-mean-manifest", type=Path)
    parser.add_argument("--new-mean-folder", type=Path)
    parser.add_argument("--new-mean-manifest", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--output-report", type=Path)
    args = parser.parse_args()
    if (
        sum(
            bool(value)
            for value in (
                args.initialize,
                args.migrate_v2 is not None,
                args.provisional_mean,
                args.qualify_previous_state is not None,
            )
        )
        > 1
    ):
        parser.error("choose exactly one special actor lifecycle mode")
    if args.initialize:
        if (
            args.state is not None
            or args.migrate_v2 is not None
            or args.candidate_folder is not None
            or args.sample_seed is not None
            or args.deterministic_mean
            or args.provisional_mean
        ):
            parser.error("initialization must be isolated")
        result = initial_state(args.parent_folder)
        report = None
    elif args.migrate_v2 is not None:
        if (
            args.state is not None
            or args.candidate_folder is not None
            or args.sample_seed is not None
            or args.deterministic_mean
            or args.provisional_mean
            or args.retention_mean_folder is None
            or args.retention_mean_manifest is None
        ):
            parser.error("migration requires isolated, authenticated mean retention evidence")
        result = migrate_v2(
            args.parent_folder,
            json.loads(args.migrate_v2.read_text(encoding="utf-8")),
            mean_folder=args.retention_mean_folder,
            mean_manifest_path=args.retention_mean_manifest,
        )
        report = None
    elif args.provisional_mean:
        if args.state is None or args.sample_seed is not None or args.candidate_folder is not None:
            parser.error("provisional mean requires only a committed provisional state")
        state = load_provisional(json.loads(args.state.read_text(encoding="utf-8")))
        result = provisional_mean_candidate(args.parent_folder, state)
        report = None
    elif args.qualify_previous_state is not None:
        if (
            args.state is None
            or args.old_parent_folder is None
            or args.old_mean_folder is None
            or args.old_mean_manifest is None
            or args.new_mean_folder is None
            or args.new_mean_manifest is None
        ):
            parser.error("qualification requires previous/provisional states and two mean audits")
        result, report = qualify_provisional(
            json.loads(args.qualify_previous_state.read_text(encoding="utf-8")),
            json.loads(args.state.read_text(encoding="utf-8")),
            old_parent_folder=args.old_parent_folder,
            old_mean_folder=args.old_mean_folder,
            old_mean_manifest=args.old_mean_manifest,
            new_parent_folder=args.parent_folder,
            new_mean_folder=args.new_mean_folder,
            new_mean_manifest=args.new_mean_manifest,
        )
    else:
        if args.state is None:
            parser.error("sampling/updating requires a committed continual state")
        if args.retention_mean_folder is not None or args.retention_mean_manifest is not None:
            parser.error("retention evidence is for migration only")
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
