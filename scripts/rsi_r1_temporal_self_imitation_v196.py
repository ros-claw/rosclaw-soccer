"""SIM_ONLY self-imitation from physically successful temporal receive rollouts."""

from __future__ import annotations

import argparse
import copy
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import torch
from rsi_r1_protected_temporal_fresh_v194 import (
    _clean,
    _pair,
    load_weights,
    run_candidate,
)
from rsi_r1_temporal_actor_critic_v193 import TemporalActorCritic, _save_weights, score
from rsi_r1_temporal_zero_v192 import TRAIN_COURSES

from rosclaw_soccer.rsi.receiving_temporal_motor_expert import TemporalMotorWeights
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.training.receiving_course_preflight import preflight_receiving_courses
from rosclaw_soccer.training.role_receiving_courses import ReceivingCourse

SCHEMA = "rosclaw_soccer.rsi.r1_temporal_self_imitation_v196.result.v1"
ALPHAS = (0.15, 0.35, 0.60)
FRESH_COURSES = (
    ReceivingCourse("red.finisher", 196001, 1.19, 0.063),
    ReceivingCourse("red.finisher", 196002, 1.29, 0.063),
    ReceivingCourse("red.finisher", 196003, 1.19, 0.0675),
    ReceivingCourse("red.finisher", 196004, 1.29, 0.0675),
    ReceivingCourse("red.finisher", 196005, 1.19, 0.0725),
    ReceivingCourse("red.finisher", 196006, 1.29, 0.0725),
    ReceivingCourse("red.finisher", 196007, 1.19, 0.077),
    ReceivingCourse("red.finisher", 196008, 1.29, 0.077),
)


def _load_actor(model: TemporalActorCritic, weights: TemporalMotorWeights) -> None:
    first, last = model.actor[0], model.actor[-1]
    assert isinstance(first, torch.nn.Linear) and isinstance(last, torch.nn.Linear)
    with torch.no_grad():
        first.weight.copy_(torch.tensor(weights.input_matrix).reshape(32, 10))
        first.bias.copy_(torch.tensor(weights.input_bias))
        last.weight.copy_(torch.tensor(weights.output_matrix).reshape(12, 32))
        last.bias.copy_(torch.tensor(weights.output_bias))


def _load_success_data(
    experience_dir: Path, experience: dict[str, Any]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[dict[str, Any]]]:
    positive = [row for row in experience["rows"] if row["strict_controlled"]]
    per_course: dict[int, int] = {}
    for row in positive:
        per_course[row["course_index"]] = per_course.get(row["course_index"], 0) + 1
    features = []
    actions = []
    sample_weights = []
    labels = []
    for row in positive:
        path = experience_dir / f"course-{row['course']['seed']}-rep-{row['repetition']}.npz"
        if hash_bytes(path.read_bytes()) != row["trajectory_hash"]:
            raise ValueError("sealed successful trajectory hash required")
        with np.load(path, allow_pickle=False) as arrays:
            if set(arrays.files) != {"frames", "features", "logits", "shaped_reward"}:
                raise ValueError("safe successful trajectory fields required")
            feature = np.asarray(arrays["features"], dtype=np.float32)
            logit = np.asarray(arrays["logits"], dtype=np.float32)
            frames = np.asarray(arrays["frames"], dtype=np.int64)
            if (
                feature.shape != (50, 10)
                or logit.shape != (50, 12)
                or not np.array_equal(frames, np.arange(15, 65))
                or not np.isfinite(feature).all()
                or not np.isfinite(logit).all()
            ):
                raise ValueError("complete finite 50 Hz successful experience required")
            features.append(feature)
            actions.append(np.tanh(logit))
            sample_weights.extend([1.0 / per_course[row["course_index"]]] * len(frames))
            labels.append(
                {
                    "course_index": row["course_index"],
                    "repetition": row["repetition"],
                    "trajectory_hash": row["trajectory_hash"],
                }
            )
    if len(labels) < 10:
        raise ValueError("enough independent exploratory successes required")
    return (
        torch.tensor(np.concatenate(features), dtype=torch.float32),
        torch.tensor(np.concatenate(actions), dtype=torch.float32),
        torch.tensor(sample_weights, dtype=torch.float32),
        labels,
    )


def _fit(
    base: TemporalActorCritic,
    states: torch.Tensor,
    target_actions: torch.Tensor,
    weights: torch.Tensor,
    anchor_states: torch.Tensor,
    alpha: float,
) -> tuple[TemporalActorCritic, dict[str, float]]:
    model = copy.deepcopy(base)
    with torch.no_grad():
        reference = base.actor(states).tanh()
        targets = (1 - alpha) * reference + alpha * target_actions
    optimizer = torch.optim.Adam(model.actor.parameters(), lr=0.0008)
    for _ in range(350):
        predicted = model.actor(states).tanh()
        behavior_loss = (weights[:, None] * (predicted - targets).square()).mean()
        anchor_loss = model.actor(anchor_states).tanh().square().mean()
        trust_loss = (predicted - reference).square().mean()
        loss = behavior_loss + 0.5 * anchor_loss + 0.15 * trust_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.actor.parameters(), 0.5)
        optimizer.step()
    return model, {
        "loss": float(loss.detach()),
        "behavior_loss": float(behavior_loss.detach()),
        "anchor_loss": float(anchor_loss.detach()),
        "trust_loss": float(trust_loss.detach()),
    }


def _evaluate(
    pool: ProcessPoolExecutor,
    asset_root: Path,
    policy_path: Path,
    courses: tuple[ReceivingCourse, ...],
    coordination: tuple[float, ...],
    left: tuple[float, ...],
    right: tuple[float, ...],
    slope: tuple[float, ...],
    weights: TemporalMotorWeights,
    protected_features: tuple[tuple[float, ...], ...],
) -> list[dict[str, Any]]:
    n = len(courses)
    return list(
        pool.map(
            run_candidate,
            (asset_root,) * n,
            (policy_path,) * n,
            courses,
            (coordination,) * n,
            (left,) * n,
            (right,) * n,
            (slope,) * n,
            (weights,) * n,
            (protected_features,) * n,
        )
    )


def train(
    asset_root: Path,
    policy_path: Path,
    parent_report: Path,
    right_report: Path,
    refine_report: Path,
    lateral_report: Path,
    zero_report_path: Path,
    v193_report_path: Path,
    v194_report_path: Path,
    experience_dir: Path,
    checkpoint: Path,
    output: Path,
) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    if output.exists() or output.resolve().is_relative_to(root):
        raise ValueError("new external SIM_ONLY self-imitation evidence required")
    fresh_bank_hash = preflight_receiving_courses(FRESH_COURSES)
    parent, right_parent, refine, lateral, zero, v193, v194, experience = (
        json.loads(path.read_text())
        for path in (
            parent_report,
            right_report,
            refine_report,
            lateral_report,
            zero_report_path,
            v193_report_path,
            v194_report_path,
            experience_dir / "report.json",
        )
    )
    if any(
        item["report_hash"] != hash_json({k: v for k, v in item.items() if k != "report_hash"})
        for item in (parent, right_parent, refine, lateral, zero, v193, v194, experience)
    ) or (
        experience["status"] != "DEVELOPMENT_TEMPORAL_EXPERIENCE_ONLY"
        or experience["v194_report_hash"] != v194["report_hash"]
        or experience["total_controlled"] != 17
        or v194["v193_report_hash"] != v193["report_hash"]
        or v193["zero_report_hash"] != zero["report_hash"]
        or parent["policy_hash"] != hash_bytes(policy_path.read_bytes())
    ):
        raise ValueError("sealed successful temporal experience lineage required")
    base_weights = load_weights(checkpoint, v193["history"][1]["checkpoint_hash"])
    states, actions, sample_weights, labels = _load_success_data(experience_dir, experience)
    anchor_states = torch.tensor(
        [feature for row in zero["rows"][-2:] for feature in row["observed_features"]],
        dtype=torch.float32,
    )
    protected_features = tuple(tuple(row["observed_features"][0]) for row in zero["rows"][-2:])
    base = TemporalActorCritic()
    _load_actor(base, base_weights)
    torch.set_num_threads(1)
    coordination = tuple(parent["selected"]["weights"])
    left = tuple(refine["best"]["left_weights"])
    right = tuple(
        float(v) for v in np.asarray(right_parent["best_right"]["weights"], dtype=np.float64) * 0.7
    )
    slope = tuple(lateral["best"]["slope"])
    sources = {
        name: hash_bytes((root / name).read_bytes())
        for name in (
            "scripts/rsi_r1_temporal_self_imitation_v196.py",
            "scripts/rsi_r1_temporal_experience_v195.py",
            "src/rosclaw_soccer/rsi/receiving_protected_temporal_expert.py",
            "src/rosclaw_soccer/training/receiving_experiment.py",
            "src/rosclaw_soccer/skills/team/independent_team_world.py",
        )
    }
    output.mkdir(parents=True)
    candidates: list[dict[str, Any]] = []
    best_rows = v194["development"]
    best_weights = base_weights
    best_stage = "v193_update_2_parent_protected"
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=4, mp_context=context) as pool:
        for alpha in ALPHAS:
            fitted, optimization = _fit(base, states, actions, sample_weights, anchor_states, alpha)
            weights = fitted.weights()
            policy_hash = _save_weights(output / f"alpha-{alpha:.2f}.npz", weights)
            rows = _evaluate(
                pool,
                asset_root,
                policy_path,
                TRAIN_COURSES,
                coordination,
                left,
                right,
                slope,
                weights,
                protected_features,
            )
            retention_equal = all(
                row["protected_episode"]
                and row["physical_trace_hash"] == old["physical_trace_hash"]
                for row, old in zip(rows[-2:], zero["rows"][-2:], strict=True)
            )
            if not retention_equal:
                raise ValueError("self-imitation violated hard parent retention")
            candidate = {
                "alpha": alpha,
                "policy_hash": policy_hash,
                "optimization": optimization,
                "score": score(rows),
                "rows": rows,
                "retention_equal": True,
            }
            candidates.append(candidate)
            (output / "progress.json").write_text(json.dumps(candidates, indent=2) + "\n")
            print(json.dumps({"alpha": alpha, "score": candidate["score"]}), flush=True)
            if score(rows) > score(best_rows):
                best_rows = rows
                best_weights = weights
                best_stage = f"alpha-{alpha:.2f}"
        candidate_rows: list[dict[str, Any]] = []
        parent_rows: list[dict[str, Any]] = []
        if best_stage != "v193_update_2_parent_protected" and score(best_rows)[1] >= 4:
            tasks = [
                (
                    asset_root,
                    policy_path,
                    course,
                    coordination,
                    left,
                    right,
                    slope,
                    best_weights,
                    protected_features,
                )
                for course in FRESH_COURSES
            ]
            for index, (candidate, old) in enumerate(pool.map(_pair, tasks)):
                candidate_rows.append(candidate)
                parent_rows.append(old)
                print(
                    json.dumps(
                        {
                            "case": index,
                            "candidate": _clean(candidate) and candidate["controlled_reception"],
                            "parent": _clean(old) and old["controlled_reception"],
                            "protected": candidate["protected_episode"],
                        }
                    ),
                    flush=True,
                )
    best_hash = _save_weights(output / "best-policy.npz", best_weights)
    candidate_pass = sum(_clean(row) and row["controlled_reception"] for row in candidate_rows)
    parent_pass = sum(_clean(row) and row["controlled_reception"] for row in parent_rows)
    qualified = (
        best_stage != "v193_update_2_parent_protected"
        and score(best_rows)[1] >= 4
        and candidate_pass == len(FRESH_COURSES)
        and candidate_pass > parent_pass
    )
    report = {
        "schema": SCHEMA,
        "experience_report_hash": experience["report_hash"],
        "source_hashes": sources,
        "partition": "SUCCESS_TRAJECTORY_IMITATION_WITH_NEW_FRESH_GATE",
        "fresh_bank_hash": fresh_bank_hash,
        "fresh_courses": [vars(course) for course in FRESH_COURSES],
        "successful_trajectory_labels": labels,
        "training_frame_labels": len(states),
        "base_score": score(v194["development"]),
        "candidates": candidates,
        "best_stage": best_stage,
        "best_score": score(best_rows),
        "best_policy_hash": best_hash,
        "fresh_candidate": candidate_rows,
        "fresh_parent": parent_rows,
        "fresh_candidate_pass": candidate_pass,
        "fresh_parent_pass": parent_pass,
        "status": "QUALIFIED_LOCAL_SELF_IMITATION_ONLY"
        if qualified
        else "REJECTED_TEMPORAL_SELF_IMITATION_GATE",
        "promotion_authorized": False,
        "video_authorized": False,
        "activation_ceiling": "SIM_ONLY",
    }
    report["report_hash"] = hash_json(report)
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if any(hash_bytes((root / name).read_bytes()) != digest for name, digest in sources.items()):
        raise ValueError("source drift during temporal self-imitation")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--parent-report", type=Path, required=True)
    parser.add_argument("--right-report", type=Path, required=True)
    parser.add_argument("--refine-report", type=Path, required=True)
    parser.add_argument("--lateral-report", type=Path, required=True)
    parser.add_argument("--zero-report", type=Path, required=True)
    parser.add_argument("--v193-report", type=Path, required=True)
    parser.add_argument("--v194-report", type=Path, required=True)
    parser.add_argument("--experience-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = train(
        args.asset_root,
        args.policy,
        args.parent_report,
        args.right_report,
        args.refine_report,
        args.lateral_report,
        args.zero_report,
        args.v193_report,
        args.v194_report,
        args.experience_dir,
        args.checkpoint,
        args.output,
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "base_score": report["base_score"],
                "best_score": report["best_score"],
                "fresh_candidate_pass": report["fresh_candidate_pass"],
                "fresh_parent_pass": report["fresh_parent_pass"],
                "report_hash": report["report_hash"],
            }
        )
    )


if __name__ == "__main__":
    main()
