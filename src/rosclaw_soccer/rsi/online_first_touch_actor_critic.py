"""SIM_ONLY online Gaussian actor-critic over authenticated Isaac contact batches.

The frozen SONIC foundation is never updated. This learner owns only six small
joint-target residuals, and no trained state has a REAL or promotion path.
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
from rosclaw_soccer.rsi.vector_first_touch_evidence import (
    audit_first_touch_candidate_execution,
    audit_vector_first_touch,
)
from rosclaw_soccer.sim.contracts import hash_json

POLICY_SIGMA_RAD = 0.02
MEAN_LIMIT_RAD = 0.04
ACTOR_LEARNING_RATE = 0.002
CRITIC_LEARNING_RATE = 0.05


def course_features(courses: tuple[tuple[float, float, float], ...]) -> np.ndarray:
    if not courses or len(set(courses)) != len(courses):
        raise ValueError("unique physical courses required")
    features = np.asarray(
        [(1.0, (x - 2.5) / 0.1, y / 0.1, vx / 0.5) for x, y, vx in courses],
        dtype=np.float64,
    )
    if (
        features.ndim != 2
        or features.shape[1] != 4
        or not np.isfinite(features).all()
        or np.max(np.abs(features)) > 1.001
    ):
        raise ValueError("first-touch course outside declared training grid")
    return features


def actor_mean(features: np.ndarray, weights: np.ndarray) -> np.ndarray:
    if weights.shape != (4, len(JOINT_NAMES)) or not np.isfinite(weights).all():
        raise ValueError("bounded actor matrix required")
    return np.asarray(
        MEAN_LIMIT_RAD * np.tanh(features @ weights / MEAN_LIMIT_RAD), dtype=np.float64
    )


def _parent_reward(row: dict[str, Any]) -> float:
    bodies = row["contact_body_indices"]
    if row["minimum_pelvis_z_m"] < 0.65:
        return -2.0
    if bodies and set(bodies) <= {0, 1}:
        return 1.0
    return -1.0 if bodies else -0.5


def _state_body(
    parent_hash: str,
    generation: int,
    actor: np.ndarray,
    critic: np.ndarray,
    sources: list[str],
) -> dict[str, Any]:
    if (
        not parent_hash.startswith("sha256:")
        or type(generation) is not int
        or generation < 0
        or actor.shape != (4, len(JOINT_NAMES))
        or critic.shape != (4,)
        or not np.isfinite(actor).all()
        or not np.isfinite(critic).all()
        or np.max(np.abs(actor)) > 0.1
        or np.max(np.abs(critic)) > 10
    ):
        raise ValueError("invalid bounded actor-critic state")
    body = {
        "schema": "rsi_isaac_online_first_touch_actor_critic_v1",
        "activation_ceiling": "SIM_ONLY",
        "promotion_authorized": False,
        "parent_report_hash": parent_hash,
        "generation": generation,
        "actor_weights": actor.tolist(),
        "critic_weights": critic.tolist(),
        "sigma_rad": POLICY_SIGMA_RAD,
        "consumed_audit_hashes": sources,
    }
    body["state_hash"] = hash_json(body)
    return body


def initial_state(parent_hash: str) -> dict[str, Any]:
    return _state_body(
        parent_hash,
        0,
        np.zeros((4, len(JOINT_NAMES)), dtype=np.float64),
        np.zeros(4, dtype=np.float64),
        [],
    )


def load_state(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    committed = {key: value for key, value in data.items() if key != "state_hash"}
    if (
        data.get("schema") != "rsi_isaac_online_first_touch_actor_critic_v1"
        or data.get("activation_ceiling") != "SIM_ONLY"
        or data.get("promotion_authorized") is not False
        or data.get("sigma_rad") != POLICY_SIGMA_RAD
        or data.get("state_hash") != hash_json(committed)
    ):
        raise ValueError("unauthenticated online actor-critic state")
    return _state_body(
        data["parent_report_hash"],
        data["generation"],
        np.asarray(data["actor_weights"], dtype=np.float64),
        np.asarray(data["critic_weights"], dtype=np.float64),
        data["consumed_audit_hashes"],
    )


def _parent_courses(parent: dict[str, Any]) -> tuple[tuple[float, float, float], ...]:
    return tuple(
        (
            row["course"]["ball_x_m"],
            row["course"]["ball_y_local_m"],
            row["course"]["ball_vx_m_s"],
        )
        for row in parent["environments"]
    )


def sample_candidate(parent_folder: Path, state: dict[str, Any], *, seed: int) -> dict[str, Any]:
    parent_audit = audit_vector_first_touch(parent_folder)
    parent = json.loads((parent_folder / "report.json").read_text(encoding="utf-8"))
    if (
        parent_audit["source_report_hash"] != state["parent_report_hash"]
        or state["state_hash"] != load_state_from_dict(state)["state_hash"]
    ):
        raise ValueError("actor state and frozen Parent are not bound")
    courses = _parent_courses(parent)
    features = course_features(courses)
    weights = np.asarray(state["actor_weights"], dtype=np.float64)
    mean = actor_mean(features, weights)
    rng = np.random.default_rng(seed)
    actions = np.clip(
        mean + rng.normal(0.0, POLICY_SIGMA_RAD, mean.shape),
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


def load_state_from_dict(data: dict[str, Any]) -> dict[str, Any]:
    committed = {key: value for key, value in data.items() if key != "state_hash"}
    if data.get("state_hash") != hash_json(committed):
        raise ValueError("online actor state digest invalid")
    return _state_body(
        data["parent_report_hash"],
        data["generation"],
        np.asarray(data["actor_weights"], dtype=np.float64),
        np.asarray(data["critic_weights"], dtype=np.float64),
        data["consumed_audit_hashes"],
    )


def update_actor_critic(
    parent_folder: Path,
    state: dict[str, Any],
    pairs: tuple[tuple[Path, Path], ...],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not pairs or len({folder for folder, _ in pairs}) != len(pairs):
        raise ValueError("nonempty distinct physics executions required")
    state = load_state_from_dict(state)
    parent_audit = audit_vector_first_touch(parent_folder)
    parent = json.loads((parent_folder / "report.json").read_text(encoding="utf-8"))
    if state["parent_report_hash"] != parent_audit["source_report_hash"]:
        raise ValueError("online update Parent mismatch")
    courses = _parent_courses(parent)
    x = course_features(courses)
    actor = np.asarray(state["actor_weights"], dtype=np.float64)
    critic = np.asarray(state["critic_weights"], dtype=np.float64)
    mean = actor_mean(x, actor)
    expected_parent = np.asarray([_parent_reward(row) for row in parent["environments"]])
    actor_gradient = np.zeros_like(actor)
    critic_gradient = np.zeros_like(critic)
    all_delta = []
    audit_hashes = []
    candidate_clean_counts = []
    for folder, manifest_path in pairs:
        audit = audit_first_touch_candidate_execution(
            folder, parent_folder=parent_folder, candidate_path=manifest_path
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        candidate = load_first_touch_candidate(
            manifest_path,
            expected_courses=courses,
            parent_report_hash=parent["report_hash"],
        )
        pre_actor_exploration = (
            state["generation"] == 0
            and not state["consumed_audit_hashes"]
            and not np.any(actor)
            and "actor_state_hash" not in manifest
        )
        if (
            not pre_actor_exploration
            and manifest.get("actor_state_hash") != state["state_hash"]
            or audit["report_hash"] in state["consumed_audit_hashes"]
            or audit["report_hash"] in audit_hashes
        ):
            raise ValueError("candidate is off-policy or already consumed")
        expected_actions = np.clip(
            mean
            + np.random.default_rng(manifest["seed"]).normal(0.0, POLICY_SIGMA_RAD, mean.shape),
            -MAX_RESIDUAL_RAD,
            MAX_RESIDUAL_RAD,
        )
        actions = np.asarray(candidate.actions_rad)
        if not np.array_equal(actions, expected_actions):
            raise ValueError("candidate actions differ from bound actor sample")
        delta = np.asarray(audit["reward_per_course"]) - expected_parent
        advantages = delta - x @ critic
        score = (actions - mean) / (POLICY_SIGMA_RAD**2)
        tanh_derivative = 1.0 - (mean / MEAN_LIMIT_RAD) ** 2
        actor_gradient += x.T @ (advantages[:, None] * score * tanh_derivative) / len(courses)
        critic_gradient += x.T @ advantages / len(courses)
        all_delta.extend(delta.tolist())
        audit_hashes.append(audit["report_hash"])
        candidate_clean_counts.append(audit["candidate_clean_foot_only_count"])
    actor_gradient /= len(pairs)
    critic_gradient /= len(pairs)
    actor_gradient -= 0.05 * actor
    norm = float(np.linalg.norm(actor_gradient))
    if norm > 5.0:
        actor_gradient *= 5.0 / norm
    next_actor = np.clip(actor + ACTOR_LEARNING_RATE * actor_gradient, -0.08, 0.08)
    next_critic = np.clip(critic + CRITIC_LEARNING_RATE * critic_gradient, -5.0, 5.0)
    next_state = _state_body(
        state["parent_report_hash"],
        state["generation"] + 1,
        next_actor,
        next_critic,
        state["consumed_audit_hashes"] + audit_hashes,
    )
    report = {
        "schema": "rsi_isaac_online_first_touch_update_v1",
        "activation_ceiling": "SIM_ONLY",
        "promotion_authorized": False,
        "parent_report_hash": state["parent_report_hash"],
        "before_state_hash": state["state_hash"],
        "after_state_hash": next_state["state_hash"],
        "physical_batch_count": len(pairs),
        "physical_episode_count": len(pairs) * len(courses),
        "audit_hashes": audit_hashes,
        "candidate_clean_counts": candidate_clean_counts,
        "parent_clean_count": parent_audit["clean_foot_only_episode_count"],
        "mean_reward_delta_vs_parent": float(np.mean(all_delta)),
        "actor_gradient_norm_before_clip": norm,
        "fresh_opened": False,
    }
    report["report_hash"] = hash_json(report)
    return next_state, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-folder", required=True, type=Path)
    parser.add_argument("--state", type=Path)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--batch-execution", action="append", type=Path, default=[])
    parser.add_argument("--batch-manifest", action="append", type=Path, default=[])
    parser.add_argument("--output-state", type=Path)
    parser.add_argument("--output-report", type=Path)
    parser.add_argument("--output-manifest", type=Path)
    args = parser.parse_args()
    parent = json.loads((args.parent_folder / "report.json").read_text(encoding="utf-8"))
    state = (
        load_state(args.state) if args.state is not None else initial_state(parent["report_hash"])
    )
    if args.output_manifest is not None:
        if args.seed is None or args.batch_execution or args.batch_manifest:
            parser.error("sampling requires seed and no update batches")
        manifest = sample_candidate(args.parent_folder, state, seed=args.seed)
        with args.output_manifest.open("x", encoding="utf-8") as stream:
            json.dump(manifest, stream, indent=2, sort_keys=True)
            stream.write("\n")
        print(
            json.dumps(
                {
                    "candidate_hash": manifest["candidate_hash"],
                    "actor_state_hash": state["state_hash"],
                }
            )
        )
        return
    if (
        not args.batch_execution
        or len(args.batch_execution) != len(args.batch_manifest)
        or args.output_state is None
        or args.output_report is None
    ):
        parser.error("update requires paired physics and manifest evidence plus output paths")
    next_state, report = update_actor_critic(
        args.parent_folder,
        state,
        tuple(zip(args.batch_execution, args.batch_manifest, strict=True)),
    )
    for path, value in ((args.output_state, next_state), (args.output_report, report)):
        with path.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
