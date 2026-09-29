"""SIM_ONLY conservative advantage-weighted 48D motor learning on consumed data."""

from __future__ import annotations

import argparse
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import torch
from rsi_r1_coherent_exploration_v198 import _checked
from rsi_r1_protected_temporal_fresh_v194 import load_weights
from rsi_r1_temporal_actor_critic_v193 import run_episode, score
from rsi_r1_temporal_zero_v192 import TRAIN_COURSES

from rosclaw_soccer.rsi.receiving_kinematic_temporal_expert import KinematicMotorWeights
from rosclaw_soccer.rsi.receiving_temporal_motor_expert import TemporalMotorWeights
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.training.receiving_course_preflight import preflight_receiving_courses

SCHEMA = "rosclaw_soccer.rsi.r1_kinematic_advantage_v201.result.v1"
ALPHAS = (0.15, 0.35)


class KinematicActor(torch.nn.Module):
    def __init__(self, weights: KinematicMotorWeights) -> None:
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(48, 32), torch.nn.Tanh(), torch.nn.Linear(32, 12)
        )
        first, last = self.network[0], self.network[2]
        assert isinstance(first, torch.nn.Linear) and isinstance(last, torch.nn.Linear)
        with torch.no_grad():
            first.weight.copy_(torch.tensor(weights.input_matrix).reshape(32, 48))
            first.bias.copy_(torch.tensor(weights.input_bias))
            last.weight.copy_(torch.tensor(weights.output_matrix).reshape(12, 32))
            last.bias.copy_(torch.tensor(weights.output_bias))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values)

    def weights(self) -> KinematicMotorWeights:
        first, last = self.network[0], self.network[2]
        assert isinstance(first, torch.nn.Linear) and isinstance(last, torch.nn.Linear)
        return KinematicMotorWeights(
            tuple(float(value) for value in first.weight.detach().reshape(-1)),
            tuple(float(value) for value in first.bias.detach()),
            tuple(float(value) for value in last.weight.detach().reshape(-1)),
            tuple(float(value) for value in last.bias.detach()),
        )


def _load_experience(
    experience_dir: Path, report: dict[str, Any]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, int]]:
    features = []
    logits = []
    advantages = []
    counts = {"strict_controlled": 0, "clean_uncontrolled": 0, "unclean": 0}
    for row in report["rows"]:
        path = experience_dir / f"course-{row['course']['seed']}-rep-{row['repetition']}.npz"
        if hash_bytes(path.read_bytes()) != row["trajectory_hash"]:
            raise ValueError("sealed 48D trajectory hash required")
        with np.load(path, allow_pickle=False) as arrays:
            if set(arrays.files) != {"frames", "features", "logits"}:
                raise ValueError("safe 48D trajectory fields required")
            frames = np.asarray(arrays["frames"], dtype=np.int64)
            state = np.asarray(arrays["features"], dtype=np.float32)
            action = np.asarray(arrays["logits"], dtype=np.float32)
        if (
            not np.array_equal(frames, np.arange(15, 65))
            or state.shape != (50, 48)
            or action.shape != (50, 12)
            or not np.isfinite(state).all()
            or not np.isfinite(action).all()
        ):
            raise ValueError("complete finite 48D trajectory required")
        features.append(state)
        logits.append(action)
        if row["strict_controlled"]:
            advantage = 1.0
            counts["strict_controlled"] += 1
        elif row["clean"]:
            advantage = 0.0
            counts["clean_uncontrolled"] += 1
        else:
            advantage = -0.5
            counts["unclean"] += 1
        advantages.extend([advantage] * 50)
    if len(features) != 128 or counts["strict_controlled"] != 17:
        raise ValueError("complete paired legacy success/failure population required")
    return (
        torch.tensor(np.concatenate(features)),
        torch.tensor(np.concatenate(logits)),
        torch.tensor(advantages, dtype=torch.float32),
        counts,
    )


def _fit(
    base_weights: KinematicMotorWeights,
    states: torch.Tensor,
    sampled_logits: torch.Tensor,
    outcomes: torch.Tensor,
    anchor_states: torch.Tensor,
    alpha: float,
) -> tuple[KinematicMotorWeights, dict[str, float]]:
    model = KinematicActor(base_weights)
    with torch.no_grad():
        reference = model(states)
        anchor_reference = model(anchor_states)
        target = reference + alpha * outcomes[:, None] * (sampled_logits - reference)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    for _ in range(400):
        predicted = model(states)
        behavior_loss = (predicted - target).square().mean()
        anchor_loss = (model(anchor_states) - anchor_reference).square().mean()
        trust_loss = (predicted - reference).square().mean()
        loss = behavior_loss + 2.0 * anchor_loss + 0.5 * trust_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
    return model.weights(), {
        "total_loss": float(loss.detach()),
        "behavior_loss": float(behavior_loss.detach()),
        "anchor_loss": float(anchor_loss.detach()),
        "trust_loss": float(trust_loss.detach()),
    }


def _run_task(task: tuple[Any, ...]) -> dict[str, Any]:
    return run_episode(*task)


def _evaluate(
    pool: ProcessPoolExecutor,
    asset_root: Path,
    policy_path: Path,
    courses: tuple[Any, ...],
    coordination: tuple[float, ...],
    left: tuple[float, ...],
    right: tuple[float, ...],
    slope: tuple[float, ...],
    weights: KinematicMotorWeights,
) -> list[dict[str, Any]]:
    return list(
        pool.map(
            _run_task,
            [
                (
                    asset_root,
                    policy_path,
                    course,
                    coordination,
                    left,
                    right,
                    slope,
                    TemporalMotorWeights(),
                    0.0,
                    0,
                    0.0,
                    weights,
                )
                for course in courses
            ],
        )
    )


def train(
    asset_root: Path,
    policy_path: Path,
    parent_report: Path,
    right_report: Path,
    refine_report: Path,
    lateral_report: Path,
    v193_report_path: Path,
    zero_report_path: Path,
    replay_dir: Path,
    checkpoint: Path,
    output: Path,
) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    if output.exists() or output.resolve().is_relative_to(root):
        raise ValueError("new external SIM_ONLY kinematic training evidence required")
    bank_hash = preflight_receiving_courses(TRAIN_COURSES)
    parent, right_parent, refine, lateral, v193, zero, replay = (
        _checked(path)
        for path in (
            parent_report,
            right_report,
            refine_report,
            lateral_report,
            v193_report_path,
            zero_report_path,
            replay_dir / "report.json",
        )
    )
    checkpoint_hash = v193["history"][1]["checkpoint_hash"]
    if (
        replay["status"] != "KINEMATIC_LEGACY_PHYSICS_EQUIVALENT"
        or replay["physical_equal_count"] != 128
        or replay["checkpoint_hash"] != checkpoint_hash
        or zero["course_bank_hash"] != bank_hash
        or parent["policy_hash"] != hash_bytes(policy_path.read_bytes())
    ):
        raise ValueError("complete physically equivalent 48D experience required")
    base = KinematicMotorWeights.from_legacy(load_weights(checkpoint, checkpoint_hash))
    states, logits, outcomes, counts = _load_experience(replay_dir, replay)
    anchors = []
    for row in zero["rows"][-2:]:
        path = Path(zero_report_path).parent / f"features-{row['course']['seed']}.npz"
        if hash_bytes(path.read_bytes()) != row["feature_hash"]:
            raise ValueError("sealed old-anchor 48D trajectory required")
        with np.load(path, allow_pickle=False) as arrays:
            anchors.append(np.asarray(arrays["features"], dtype=np.float32))
    anchor_states = torch.tensor(np.concatenate(anchors))
    torch.set_num_threads(1)
    torch.manual_seed(201001)
    coordination = tuple(parent["selected"]["weights"])
    left = tuple(refine["best"]["left_weights"])
    right = tuple(float(value) for value in np.asarray(right_parent["best_right"]["weights"]) * 0.7)
    slope = tuple(lateral["best"]["slope"])
    sources = {
        name: hash_bytes((root / name).read_bytes())
        for name in (
            "scripts/rsi_r1_kinematic_advantage_v201.py",
            "scripts/rsi_r1_temporal_actor_critic_v193.py",
            "src/rosclaw_soccer/rsi/receiving_kinematic_temporal_expert.py",
            "src/rosclaw_soccer/training/receiving_experiment.py",
            "src/rosclaw_soccer/skills/team/independent_team_world.py",
        )
    }
    output.mkdir(parents=True)
    candidates = []
    with ProcessPoolExecutor(
        max_workers=4, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        baseline = _evaluate(
            pool, asset_root, policy_path, TRAIN_COURSES, coordination, left, right, slope, base
        )
        old_score = v193["history"][1]["deterministic_score"]
        baseline_score = score(baseline)
        if (
            tuple(baseline_score[:3]) != tuple(old_score[:3])
            or abs(baseline_score[3] - old_score[3]) > 1e-6
        ):
            raise ValueError("deterministic embedded legacy policy drift")
        for alpha in ALPHAS:
            fitted, loss = _fit(base, states, logits, outcomes, anchor_states, alpha)
            policy_path_out = output / f"alpha-{alpha:.2f}.npz"
            np.savez_compressed(
                policy_path_out,
                input_matrix=np.asarray(fitted.input_matrix),
                input_bias=np.asarray(fitted.input_bias),
                output_matrix=np.asarray(fitted.output_matrix),
                output_bias=np.asarray(fitted.output_bias),
            )
            rows = _evaluate(
                pool,
                asset_root,
                policy_path,
                TRAIN_COURSES,
                coordination,
                left,
                right,
                slope,
                fitted,
            )
            candidates.append(
                {
                    "alpha": alpha,
                    "policy_hash": hash_bytes(policy_path_out.read_bytes()),
                    "loss": loss,
                    "score": score(rows),
                    "rows": rows,
                }
            )
            (output / "progress.json").write_text(json.dumps(candidates, indent=2) + "\n")
            print(json.dumps({"alpha": alpha, "score": score(rows)}), flush=True)
    baseline_score = score(baseline)
    best = max(candidates, key=lambda row: tuple(row["score"]))
    qualified = best["score"][0] == 1 and best["score"][1] >= 4 and best["score"] > baseline_score
    report = {
        "schema": SCHEMA,
        "replay_report_hash": replay["report_hash"],
        "source_hashes": sources,
        "partition": "CONSUMED_EIGHT_G1_KINEMATIC_ADVANTAGE_TRAINING_NOT_FRESH",
        "course_bank_hash": bank_hash,
        "outcome_counts": counts,
        "training_frames": len(states),
        "baseline_score": baseline_score,
        "candidates": candidates,
        "best_alpha": best["alpha"] if qualified else None,
        "status": "DEVELOPMENT_KINEMATIC_CANDIDATE_ONLY"
        if qualified
        else "REJECTED_KINEMATIC_ADVANTAGE_GATE",
        "promotion_authorized": False,
        "video_authorized": False,
        "activation_ceiling": "SIM_ONLY",
    }
    report["report_hash"] = hash_json(report)
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if any(hash_bytes((root / name).read_bytes()) != digest for name, digest in sources.items()):
        raise ValueError("source drift during kinematic advantage training")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in (
        "asset-root",
        "policy",
        "parent-report",
        "right-report",
        "refine-report",
        "lateral-report",
        "v193-report",
        "zero-report",
        "replay-dir",
        "checkpoint",
        "output",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    report = train(
        args.asset_root,
        args.policy,
        args.parent_report,
        args.right_report,
        args.refine_report,
        args.lateral_report,
        args.v193_report,
        args.zero_report,
        args.replay_dir,
        args.checkpoint,
        args.output,
    )
    print(
        json.dumps(
            {key: report[key] for key in ("status", "baseline_score", "best_alpha", "report_hash")}
        )
    )


if __name__ == "__main__":
    main()
