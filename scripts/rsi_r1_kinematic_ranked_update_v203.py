"""SIM_ONLY course-relative actor update using measured contact and ball quality."""

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
from rsi_r1_kinematic_advantage_v201 import KinematicActor
from rsi_r1_protected_kinematic_fresh_v202 import _load_policy, _run_task, _tasks
from rsi_r1_temporal_actor_critic_v193 import score
from rsi_r1_temporal_zero_v192 import TRAIN_COURSES

from rosclaw_soccer.rsi.receiving_kinematic_temporal_expert import KinematicMotorWeights
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.training.receiving_course_preflight import preflight_receiving_courses

SCHEMA = "rosclaw_soccer.rsi.r1_kinematic_ranked_update_v203.result.v1"
ALPHAS = (0.10, 0.25)


def _quality(row: dict[str, Any]) -> float:
    """Safety-first continuous quality; never treat a nonfoot collision as success."""
    if not row["safe"] or row["fault_agents"]:
        return -3.0
    if not row["clean"]:
        return -1.5 - 0.1 * min(len(row["own_nonfoot_frames"]), 10)
    distance = min(float(row["tail_maximum_foot_distance_m"]), 2.0)
    speed = min(float(row["tail_maximum_ball_speed_mps"]), 3.0)
    return 1.0 + 2.0 * float(row["strict_controlled"]) - 0.4 * distance - 0.25 * speed


def _load_ranked(
    replay_dir: Path, replay: dict[str, Any], experience: dict[str, Any]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[dict[str, float]]]:
    states = []
    actions = []
    advantages = []
    course_stats = []
    for course_index in range(16):
        rows = replay["rows"][course_index * 8 : (course_index + 1) * 8]
        old_rows = experience["rows"][course_index * 8 : (course_index + 1) * 8]
        if len(rows) != 8 or len(old_rows) != 8:
            raise ValueError("eight paired repetitions per consumed course required")
        quality = np.asarray([_quality(row) for row in old_rows], dtype=np.float64)
        centered = quality - quality.mean()
        scale = max(float(quality.std()), 0.25)
        ranked = np.clip(centered / scale, -2.0, 2.0)
        course_stats.append(
            {
                "course_index": float(course_index),
                "clean_count": float(sum(row["clean"] for row in old_rows)),
                "controlled_count": float(sum(row["strict_controlled"] for row in old_rows)),
                "quality_std": float(quality.std()),
            }
        )
        for repetition, (row, old) in enumerate(zip(rows, old_rows, strict=True)):
            if (
                row["course"] != old["course"]
                or row["repetition"] != old["repetition"]
                or not row["physical_equal"]
                or row["physical_trace_hash"] != old["physical_trace_hash"]
            ):
                raise ValueError("paired physically equal training row required")
            path = replay_dir / f"course-{row['course']['seed']}-rep-{repetition}.npz"
            if hash_bytes(path.read_bytes()) != row["trajectory_hash"]:
                raise ValueError("sealed 48D ranked trajectory required")
            with np.load(path, allow_pickle=False) as arrays:
                if set(arrays.files) != {"frames", "features", "logits"}:
                    raise ValueError("safe ranked trajectory arrays required")
                frames = np.asarray(arrays["frames"], dtype=np.int64)
                features = np.asarray(arrays["features"], dtype=np.float32)
                logits = np.asarray(arrays["logits"], dtype=np.float32)
            if (
                not np.array_equal(frames, np.arange(15, 65))
                or features.shape != (50, 48)
                or logits.shape != (50, 12)
                or not np.isfinite(features).all()
                or not np.isfinite(logits).all()
            ):
                raise ValueError("complete finite 48D ranked trajectory required")
            states.append(features)
            actions.append(logits)
            advantages.extend([float(ranked[repetition])] * 50)
    return (
        torch.tensor(np.concatenate(states)),
        torch.tensor(np.concatenate(actions)),
        torch.tensor(advantages, dtype=torch.float32),
        course_stats,
    )


def _fit(
    base: KinematicMotorWeights,
    states: torch.Tensor,
    sampled: torch.Tensor,
    advantage: torch.Tensor,
    alpha: float,
) -> tuple[KinematicMotorWeights, dict[str, float]]:
    model = KinematicActor(base)
    with torch.no_grad():
        reference = model(states)
        target = reference + alpha * advantage[:, None] * (sampled - reference)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00035)
    for _ in range(400):
        predicted = model(states)
        behavior_loss = (predicted - target).square().mean()
        trust_loss = (predicted - reference).square().mean()
        loss = behavior_loss + 0.7 * trust_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
    return model.weights(), {
        "loss": float(loss.detach()),
        "behavior_loss": float(behavior_loss.detach()),
        "trust_loss": float(trust_loss.detach()),
    }


def train(
    asset_root: Path,
    policy_path: Path,
    parent_report: Path,
    right_report: Path,
    refine_report: Path,
    lateral_report: Path,
    v195_report_path: Path,
    replay_dir: Path,
    zero_report_path: Path,
    v202_report_path: Path,
    base_policy_path: Path,
    output: Path,
) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    if output.exists() or output.resolve().is_relative_to(root):
        raise ValueError("new external SIM_ONLY ranked training evidence required")
    bank_hash = preflight_receiving_courses(TRAIN_COURSES)
    parent, right_parent, refine, lateral, experience, replay, zero, v202 = (
        _checked(path)
        for path in (
            parent_report,
            right_report,
            refine_report,
            lateral_report,
            v195_report_path,
            replay_dir / "report.json",
            zero_report_path,
            v202_report_path,
        )
    )
    if (
        v202["status"] != "REJECTED_PROTECTED_KINEMATIC_FRESH_GATE"
        or not v202["retention_physical_equal"]
        or v202["development_score"][:3] != [1, 4, 10]
        or replay["status"] != "KINEMATIC_LEGACY_PHYSICS_EQUIVALENT"
        or replay["experience_report_hash"] != experience["report_hash"]
        or zero["course_bank_hash"] != bank_hash
        or parent["policy_hash"] != hash_bytes(policy_path.read_bytes())
    ):
        raise ValueError("sealed rejected 48D exam and paired online experience required")
    base = _load_policy(base_policy_path, v202["learned_policy_hash"])
    states, sampled, advantages, stats = _load_ranked(replay_dir, replay, experience)
    protected = tuple(
        tuple(float(value) for value in row["observed_features"][0]) for row in zero["rows"][-2:]
    )
    coordination = tuple(parent["selected"]["weights"])
    left = tuple(refine["best"]["left_weights"])
    right = tuple(float(value) for value in np.asarray(right_parent["best_right"]["weights"]) * 0.7)
    slope = tuple(lateral["best"]["slope"])
    sources = {
        name: hash_bytes((root / name).read_bytes())
        for name in (
            "scripts/rsi_r1_kinematic_ranked_update_v203.py",
            "scripts/rsi_r1_kinematic_advantage_v201.py",
            "scripts/rsi_r1_protected_kinematic_fresh_v202.py",
            "scripts/rsi_r1_temporal_actor_critic_v193.py",
            "src/rosclaw_soccer/rsi/receiving_kinematic_temporal_expert.py",
            "src/rosclaw_soccer/training/receiving_experiment.py",
            "src/rosclaw_soccer/skills/team/independent_team_world.py",
        )
    }
    torch.manual_seed(203001)
    torch.set_num_threads(1)
    output.mkdir(parents=True)
    candidates = []
    with ProcessPoolExecutor(
        max_workers=4, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        for alpha in ALPHAS:
            fitted, optimization = _fit(base, states, sampled, advantages, alpha)
            policy_out = output / f"alpha-{alpha:.2f}.npz"
            np.savez_compressed(
                policy_out,
                input_matrix=np.asarray(fitted.input_matrix),
                input_bias=np.asarray(fitted.input_bias),
                output_matrix=np.asarray(fitted.output_matrix),
                output_bias=np.asarray(fitted.output_bias),
            )
            rows = list(
                pool.map(
                    _run_task,
                    _tasks(
                        asset_root,
                        policy_path,
                        TRAIN_COURSES,
                        coordination,
                        left,
                        right,
                        slope,
                        fitted,
                        protected,
                    ),
                )
            )
            retention_equal = all(
                row["protected_episode"] is True
                and row["physical_trace_hash"] == old["physical_trace_hash"]
                for row, old in zip(rows[-2:], zero["rows"][-2:], strict=True)
            )
            if not retention_equal:
                raise ValueError("ranked update violated hard old-skill protection")
            candidates.append(
                {
                    "alpha": alpha,
                    "policy_hash": hash_bytes(policy_out.read_bytes()),
                    "optimization": optimization,
                    "score": score(rows),
                    "rows": rows,
                    "retention_physical_equal": True,
                }
            )
            (output / "progress.json").write_text(json.dumps(candidates, indent=2) + "\n")
            print(json.dumps({"alpha": alpha, "score": score(rows)}), flush=True)
    best = max(candidates, key=lambda row: tuple(row["score"]))
    qualified = tuple(best["score"]) > tuple(v202["development_score"]) and best["score"][1] >= 5
    report = {
        "schema": SCHEMA,
        "v202_report_hash": v202["report_hash"],
        "source_hashes": sources,
        "partition": "RANKED_UPDATE_ON_CONSUMED_EXPERIENCE_NOT_FRESH",
        "course_bank_hash": bank_hash,
        "course_quality_stats": stats,
        "training_frames": len(states),
        "baseline_score": v202["development_score"],
        "candidates": candidates,
        "best_alpha": best["alpha"] if qualified else None,
        "status": "DEVELOPMENT_RANKED_CANDIDATE_ONLY"
        if qualified
        else "REJECTED_RANKED_UPDATE_GATE",
        "promotion_authorized": False,
        "video_authorized": False,
        "activation_ceiling": "SIM_ONLY",
    }
    report["report_hash"] = hash_json(report)
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if any(hash_bytes((root / name).read_bytes()) != digest for name, digest in sources.items()):
        raise ValueError("source drift during ranked kinematic update")
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
        "v195-report",
        "replay-dir",
        "zero-report",
        "v202-report",
        "base-policy",
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
        args.v195_report,
        args.replay_dir,
        args.zero_report,
        args.v202_report,
        args.base_policy,
        args.output,
    )
    print(
        json.dumps(
            {key: report[key] for key in ("status", "baseline_score", "best_alpha", "report_hash")}
        )
    )


if __name__ == "__main__":
    main()
