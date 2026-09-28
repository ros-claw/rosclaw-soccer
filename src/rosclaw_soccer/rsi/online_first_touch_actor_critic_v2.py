"""SIM_ONLY contextual actor-critic for disjoint sixteen-course Isaac batches.

The frozen SONIC backbone never changes. This learner only samples bounded
six-joint precontact residuals and requires authenticated physical feedback.
No output carries REAL authority or policy promotion.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.rsi.first_touch_candidate import (
    JOINT_NAMES,
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

SCHEMA = "rsi_isaac_online_first_touch_actor_critic_v2"
SIGMA_RAD = 0.05
MEAN_LIMIT_RAD = 0.04
ACTOR_STEP = 0.002
CRITIC_STEP = 0.05
MAX_GENERATION_MEAN_SHIFT_RAD = 0.01
STATE_FIELDS = frozenset(
    {
        "schema",
        "activation_ceiling",
        "promotion_authorized",
        "parent_report_hash",
        "training_course_seed",
        "course_catalog_hash",
        "generation",
        "actor_weights",
        "critic_weights",
        "sigma_rad",
        "consumed_audit_hashes",
        "state_hash",
    }
)


def _parent(folder: Path) -> tuple[dict[str, Any], tuple[tuple[float, float, float], ...]]:
    audited = audit_vector_first_touch(folder)
    report = json.loads((folder / "report.json").read_text(encoding="utf-8"))
    seed = report.get("training_course_seed")
    if (
        audited["source_report_hash"] != report["report_hash"]
        or type(seed) is not int
        or len(report["environments"]) != 16
        or report.get("navigation_speed_mps", 1.4) != 1.4
        or "near_ball_gap_m" in report
    ):
        raise ValueError("v2 learner requires authenticated seeded sixteen-course Parent")
    courses = sample_training_courses(seed)
    if report.get("course_catalog_hash") != hash_json(courses):
        raise ValueError("Parent training split changed")
    return report, courses


def _features(courses: tuple[tuple[float, float, float], ...]) -> np.ndarray:
    x = np.asarray(
        [(1.0, (p - 2.5) / 0.3, y / 0.16, velocity / 0.7) for p, y, velocity in courses],
        dtype=np.float64,
    )
    if x.shape != (16, 4) or not np.isfinite(x).all() or np.max(np.abs(x)) > 1.001:
        raise ValueError("training course feature envelope changed")
    return x


def _mean(features: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return np.asarray(MEAN_LIMIT_RAD * np.tanh(features @ weights / MEAN_LIMIT_RAD))


def _state(
    parent: dict[str, Any],
    generation: int,
    actor: np.ndarray,
    critic: np.ndarray,
    consumed: list[str],
) -> dict[str, Any]:
    if (
        type(generation) is not int
        or generation < 0
        or actor.shape != (4, len(JOINT_NAMES))
        or critic.shape != (4,)
        or not np.isfinite(actor).all()
        or not np.isfinite(critic).all()
        or np.max(np.abs(actor)) > 0.1
        or np.max(np.abs(critic)) > 10
        or not isinstance(consumed, list)
        or len(consumed) != len(set(consumed))
        or any(not isinstance(value, str) or not value.startswith("sha256:") for value in consumed)
    ):
        raise ValueError("invalid bounded v2 actor-critic state")
    body: dict[str, Any] = {
        "schema": SCHEMA,
        "activation_ceiling": "SIM_ONLY",
        "promotion_authorized": False,
        "parent_report_hash": parent["report_hash"],
        "training_course_seed": parent["training_course_seed"],
        "course_catalog_hash": parent["course_catalog_hash"],
        "generation": generation,
        "actor_weights": actor.tolist(),
        "critic_weights": critic.tolist(),
        "sigma_rad": SIGMA_RAD,
        "consumed_audit_hashes": consumed,
    }
    body["state_hash"] = hash_json(body)
    return body


def initial_state(parent_folder: Path) -> dict[str, Any]:
    parent, _ = _parent(parent_folder)
    return _state(parent, 0, np.zeros((4, 6)), np.zeros(4), [])


def load_state_from_dict(parent: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    committed = {key: value for key, value in state.items() if key != "state_hash"}
    if (
        set(state) != STATE_FIELDS
        or state.get("schema") != SCHEMA
        or state.get("activation_ceiling") != "SIM_ONLY"
        or state.get("promotion_authorized") is not False
        or state.get("sigma_rad") != SIGMA_RAD
        or state.get("parent_report_hash") != parent["report_hash"]
        or state.get("training_course_seed") != parent["training_course_seed"]
        or state.get("course_catalog_hash") != parent["course_catalog_hash"]
        or state.get("state_hash") != hash_json(committed)
    ):
        raise ValueError("v2 actor state commitment or Parent binding invalid")
    canonical = _state(
        parent,
        state["generation"],
        np.asarray(state["actor_weights"], dtype=np.float64),
        np.asarray(state["critic_weights"], dtype=np.float64),
        state["consumed_audit_hashes"],
    )
    if canonical["state_hash"] != state["state_hash"]:
        raise ValueError("v2 actor state canonical form changed")
    return canonical


def sample_candidate(parent_folder: Path, state: dict[str, Any], *, seed: int) -> dict[str, Any]:
    parent, courses = _parent(parent_folder)
    state = load_state_from_dict(parent, state)
    mean = _mean(_features(courses), np.asarray(state["actor_weights"]))
    rng = np.random.default_rng(seed)
    actions = np.clip(
        mean + rng.normal(0.0, SIGMA_RAD, mean.shape), -MAX_RESIDUAL_RAD, MAX_RESIDUAL_RAD
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
    """Evaluate the learned mean separately from stochastic exploration."""
    parent, courses = _parent(parent_folder)
    state = load_state_from_dict(parent, state)
    mean = _mean(_features(courses), np.asarray(state["actor_weights"]))
    manifest = candidate_manifest(
        courses=courses,
        parent_report_hash=parent["report_hash"],
        actions_rad=tuple(tuple(float(value) for value in row) for row in mean),
        seed=0,
    )
    manifest["actor_state_hash"] = state["state_hash"]
    manifest["evaluation_mode"] = "FROZEN_ACTOR_MEAN"
    manifest.pop("candidate_hash")
    manifest["candidate_hash"] = hash_json(manifest)
    return manifest


def _parent_reward(row: dict[str, Any]) -> float:
    bodies = row["contact_body_indices"]
    if row["minimum_pelvis_z_m"] < 0.65:
        return -2.0
    return 1.0 if bodies and set(bodies) <= {0, 1} else -1.0 if bodies else -0.5


def update_actor_critic(
    parent_folder: Path,
    state: dict[str, Any],
    pairs: tuple[tuple[Path, Path], ...],
) -> tuple[dict[str, Any], dict[str, Any]]:
    parent, courses = _parent(parent_folder)
    state = load_state_from_dict(parent, state)
    if not pairs or len({folder for folder, _ in pairs}) != len(pairs):
        raise ValueError("distinct authenticated physics batches required")
    x = _features(courses)
    actor = np.asarray(state["actor_weights"], dtype=np.float64)
    critic = np.asarray(state["critic_weights"], dtype=np.float64)
    mean = _mean(x, actor)
    parent_reward = np.asarray([_parent_reward(row) for row in parent["environments"]])
    actor_gradient = np.zeros_like(actor)
    critic_gradient = np.zeros_like(critic)
    audit_hashes: list[str] = []
    deltas: list[float] = []
    clean_counts: list[int] = []
    for folder, manifest_path in pairs:
        audit = audit_first_touch_candidate_execution(
            folder, parent_folder=parent_folder, candidate_path=manifest_path
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        candidate = load_first_touch_candidate(
            manifest_path, expected_courses=courses, parent_report_hash=parent["report_hash"]
        )
        initial_unbound = (
            state["generation"] == 0
            and not state["consumed_audit_hashes"]
            and not np.any(actor)
            and "actor_state_hash" not in manifest
        )
        if (
            not initial_unbound
            and manifest.get("actor_state_hash") != state["state_hash"]
            or audit["report_hash"] in state["consumed_audit_hashes"]
            or audit["report_hash"] in audit_hashes
        ):
            raise ValueError("candidate is off-policy or physical batch already consumed")
        expected = np.clip(
            mean + np.random.default_rng(manifest["seed"]).normal(0.0, SIGMA_RAD, mean.shape),
            -MAX_RESIDUAL_RAD,
            MAX_RESIDUAL_RAD,
        )
        actions = np.asarray(candidate.actions_rad)
        if not np.array_equal(expected, actions):
            raise ValueError("candidate actions differ from bound actor sample")
        delta = np.asarray(audit["reward_per_course"]) - parent_reward
        advantage = delta - x @ critic
        score = (actions - mean) / SIGMA_RAD**2
        derivative = 1.0 - (mean / MEAN_LIMIT_RAD) ** 2
        actor_gradient += x.T @ (advantage[:, None] * score * derivative) / len(courses)
        critic_gradient += x.T @ advantage / len(courses)
        audit_hashes.append(audit["report_hash"])
        deltas.extend(delta.tolist())
        clean_counts.append(audit["candidate_clean_foot_only_count"])
    actor_gradient = actor_gradient / len(pairs) - 0.05 * actor
    critic_gradient /= len(pairs)
    gradient_norm = float(np.linalg.norm(actor_gradient))
    if gradient_norm > 5:
        actor_gradient *= 5 / gradient_norm
    proposal = np.clip(actor + ACTOR_STEP * actor_gradient, -0.08, 0.08)
    proposed_mean = _mean(x, proposal)
    drift = float(np.max(np.abs(proposed_mean - mean)))
    if drift > MAX_GENERATION_MEAN_SHIFT_RAD:
        lower, upper = 0.0, 1.0
        direction = proposal - actor
        for _ in range(32):
            middle = (lower + upper) / 2
            trial_actor = actor + middle * direction
            candidate_drift = float(np.max(np.abs(_mean(x, trial_actor) - mean)))
            if candidate_drift <= MAX_GENERATION_MEAN_SHIFT_RAD:
                lower = middle
            else:
                upper = middle
        proposal = actor + lower * direction
        proposed_mean = _mean(x, proposal)
    actual_drift = float(np.max(np.abs(proposed_mean - mean)))
    if actual_drift > MAX_GENERATION_MEAN_SHIFT_RAD + 1e-9:
        raise ValueError("stability-plasticity trust region failed")
    next_state = _state(
        parent,
        state["generation"] + 1,
        proposal,
        np.clip(critic + CRITIC_STEP * critic_gradient, -5, 5),
        state["consumed_audit_hashes"] + audit_hashes,
    )
    report: dict[str, Any] = {
        "schema": "rsi_isaac_online_first_touch_update_v2",
        "activation_ceiling": "SIM_ONLY",
        "promotion_authorized": False,
        "parent_report_hash": parent["report_hash"],
        "before_state_hash": state["state_hash"],
        "after_state_hash": next_state["state_hash"],
        "physical_episode_count": len(pairs) * len(courses),
        "audit_hashes": audit_hashes,
        "parent_clean_count": int(np.sum(parent_reward == 1.0)),
        "candidate_clean_counts": clean_counts,
        "mean_reward_delta_vs_parent": float(np.mean(deltas)),
        "actor_gradient_norm_before_clip": gradient_norm,
        "maximum_actor_mean_shift_rad": actual_drift,
        "fresh_opened": False,
    }
    report["report_hash"] = hash_json(report)
    return next_state, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-folder", required=True, type=Path)
    parser.add_argument("--state", type=Path)
    parser.add_argument("--sample-seed", type=int)
    parser.add_argument("--deterministic-mean", action="store_true")
    parser.add_argument("--candidate-folder", type=Path)
    parser.add_argument("--candidate-manifest", type=Path)
    parser.add_argument("--output-manifest", type=Path)
    parser.add_argument("--output-state", type=Path)
    parser.add_argument("--output-report", type=Path)
    args = parser.parse_args()
    parent, _ = _parent(args.parent_folder)
    state = (
        load_state_from_dict(parent, json.loads(args.state.read_text(encoding="utf-8")))
        if args.state is not None
        else initial_state(args.parent_folder)
    )
    if args.output_manifest is not None:
        if args.candidate_folder is not None or (args.sample_seed is None) == (
            not args.deterministic_mean
        ):
            parser.error("sampling requires exactly one stochastic or mean mode")
        result = (
            deterministic_mean_candidate(args.parent_folder, state)
            if args.deterministic_mean
            else sample_candidate(args.parent_folder, state, seed=args.sample_seed)
        )
        with args.output_manifest.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2, sort_keys=True)
            stream.write("\n")
        print(json.dumps({"candidate_hash": result["candidate_hash"]}))
        return
    if (
        args.candidate_folder is None
        or args.candidate_manifest is None
        or args.output_state is None
        or args.output_report is None
        or args.sample_seed is not None
        or args.deterministic_mean
    ):
        parser.error("update requires paired physics evidence and output paths")
    next_state, report = update_actor_critic(
        args.parent_folder, state, ((args.candidate_folder, args.candidate_manifest),)
    )
    for path, value in ((args.output_state, next_state), (args.output_report, report)):
        with path.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
