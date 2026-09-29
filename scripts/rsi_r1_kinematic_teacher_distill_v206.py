"""SIM_ONLY distillation of sealed real-contact specialist trajectories into 48D actor."""

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
from rsi_r1_temporal_actor_critic_v193 import clean, score
from rsi_r1_temporal_zero_v192 import TRAIN_COURSES

from rosclaw_soccer.rsi.receiving_kinematic_temporal_expert import KinematicMotorWeights
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.training.receiving_course_preflight import preflight_receiving_courses

SCHEMA = "rosclaw_soccer.rsi.r1_kinematic_teacher_distill_v206.result.v1"
ALPHAS = (0.20, 0.50)
HIGH_SPEED_SEEDS = (190008, 190011)


def _load_teachers(
    teacher_dir: Path, report: dict[str, Any]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    states = []
    targets = []
    weights = []
    for index, row in enumerate(report["rows"]):
        path = teacher_dir / f"teacher-{index}.npz"
        if hash_bytes(path.read_bytes()) != row["feature_hash"] or not row["physical_equal"]:
            raise ValueError("sealed physically equal specialist teacher required")
        with np.load(path, allow_pickle=False) as arrays:
            if set(arrays.files) != {"frames", "features", "target_actions"}:
                raise ValueError("safe teacher arrays required")
            frames = np.asarray(arrays["frames"], dtype=np.int64)
            features = np.asarray(arrays["features"], dtype=np.float32)
            actions = np.asarray(arrays["target_actions"], dtype=np.float32)
        if (
            not np.array_equal(frames, np.arange(15, 65))
            or features.shape != (50, 48)
            or actions.shape != (50, 12)
            or not np.isfinite(features).all()
            or not np.isfinite(actions).all()
        ):
            raise ValueError("complete finite 48D teacher trajectory required")
        states.append(features)
        targets.append(actions)
        weight = 3.0 if row["course"]["speed_mps"] >= 1.28 else 1.0
        weights.extend([weight] * 50)
    if len(states) != 7 or sum(row["course"]["speed_mps"] >= 1.28 for row in report["rows"]) != 2:
        raise ValueError("seven genuine successes and two high-speed teachers required")
    return (
        torch.tensor(np.concatenate(states)),
        torch.tensor(np.concatenate(targets)),
        torch.tensor(weights, dtype=torch.float32),
    )


def _anchors(zero_dir: Path, report: dict[str, Any]) -> torch.Tensor:
    arrays = []
    for row in report["rows"][-2:]:
        path = zero_dir / f"features-{row['course']['seed']}.npz"
        if hash_bytes(path.read_bytes()) != row["feature_hash"]:
            raise ValueError("sealed old-skill body states required")
        with np.load(path, allow_pickle=False) as data:
            if set(data.files) != {"features"}:
                raise ValueError("safe old-skill body states required")
            values = np.asarray(data["features"], dtype=np.float32)
        if values.shape != (50, 48) or not np.isfinite(values).all():
            raise ValueError("complete old-skill body states required")
        arrays.append(values)
    return torch.tensor(np.concatenate(arrays))


def _fit(
    base: KinematicMotorWeights,
    states: torch.Tensor,
    actions: torch.Tensor,
    weights: torch.Tensor,
    anchors: torch.Tensor,
    alpha: float,
) -> tuple[KinematicMotorWeights, dict[str, float]]:
    model = KinematicActor(base)
    with torch.no_grad():
        reference = model(states).tanh()
        anchor_reference = model(anchors).tanh()
        target = (1.0 - alpha) * reference + alpha * actions
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0004)
    for _ in range(350):
        predicted = model(states).tanh()
        imitation = (weights[:, None] * (predicted - target).square()).mean()
        anchor_loss = (model(anchors).tanh() - anchor_reference).square().mean()
        trust_loss = (predicted - reference).square().mean()
        loss = imitation + 0.7 * anchor_loss + 0.25 * trust_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
    return model.weights(), {
        "loss": float(loss.detach()),
        "imitation_loss": float(imitation.detach()),
        "anchor_loss": float(anchor_loss.detach()),
        "trust_loss": float(trust_loss.detach()),
    }


def train(
    asset_root: Path,
    policy_path: Path,
    parent_report: Path,
    right_report: Path,
    refine_report: Path,
    lateral_report: Path,
    old_zero_report: Path,
    kinematic_zero_dir: Path,
    v202_report: Path,
    teacher_dir: Path,
    base_policy_path: Path,
    output: Path,
) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    if output.exists() or output.resolve().is_relative_to(root):
        raise ValueError("new external SIM_ONLY teacher distillation evidence required")
    bank_hash = preflight_receiving_courses(TRAIN_COURSES)
    parent, right_parent, refine, lateral, old, zero, v202, teachers = (
        _checked(path)
        for path in (
            parent_report,
            right_report,
            refine_report,
            lateral_report,
            old_zero_report,
            kinematic_zero_dir / "report.json",
            v202_report,
            teacher_dir / "report.json",
        )
    )
    if (
        teachers["status"] != "PHYSICALLY_VERIFIED_48D_TEACHERS_ONLY"
        or teachers["physical_equal_count"] != 7
        or zero["course_bank_hash"] != bank_hash
        or v202["development_score"][:3] != [1, 4, 10]
        or parent["policy_hash"] != hash_bytes(policy_path.read_bytes())
    ):
        raise ValueError("sealed genuine specialist and old-skill lineage required")
    base = _load_policy(base_policy_path, v202["learned_policy_hash"])
    states, actions, sample_weights = _load_teachers(teacher_dir, teachers)
    anchors = _anchors(kinematic_zero_dir, zero)
    protected = tuple(
        tuple(float(value) for value in row["observed_features"][0]) for row in old["rows"][-2:]
    )
    coordination = tuple(parent["selected"]["weights"])
    left = tuple(refine["best"]["left_weights"])
    right = tuple(float(value) for value in np.asarray(right_parent["best_right"]["weights"]) * 0.7)
    slope = tuple(lateral["best"]["slope"])
    sources = {
        name: hash_bytes((root / name).read_bytes())
        for name in (
            "scripts/rsi_r1_kinematic_teacher_distill_v206.py",
            "scripts/rsi_r1_kinematic_teacher_capture_v205.py",
            "scripts/rsi_r1_kinematic_advantage_v201.py",
            "scripts/rsi_r1_protected_kinematic_fresh_v202.py",
            "scripts/rsi_r1_temporal_actor_critic_v193.py",
            "src/rosclaw_soccer/rsi/receiving_kinematic_temporal_expert.py",
            "src/rosclaw_soccer/training/receiving_experiment.py",
            "src/rosclaw_soccer/skills/team/independent_team_world.py",
        )
    }
    torch.manual_seed(206001)
    torch.set_num_threads(1)
    output.mkdir(parents=True)
    candidates = []
    with ProcessPoolExecutor(
        max_workers=4, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        for alpha in ALPHAS:
            fitted, optimization = _fit(base, states, actions, sample_weights, anchors, alpha)
            path = output / f"alpha-{alpha:.2f}.npz"
            np.savez_compressed(
                path,
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
                and row["physical_trace_hash"] == anchor["physical_trace_hash"]
                for row, anchor in zip(rows[-2:], old["rows"][-2:], strict=True)
            )
            if not retention_equal:
                raise ValueError("teacher distillation violated old-skill protection")
            high_speed_pass = sum(
                clean(row) and row["controlled_reception"]
                for row in rows[:16]
                if row["course"]["seed"] in HIGH_SPEED_SEEDS
            )
            candidates.append(
                {
                    "alpha": alpha,
                    "policy_hash": hash_bytes(path.read_bytes()),
                    "optimization": optimization,
                    "score": score(rows),
                    "high_speed_teacher_course_pass": high_speed_pass,
                    "rows": rows,
                    "retention_physical_equal": True,
                }
            )
            (output / "progress.json").write_text(json.dumps(candidates, indent=2) + "\n")
            print(
                json.dumps({"alpha": alpha, "score": score(rows), "high_speed": high_speed_pass}),
                flush=True,
            )
    best = max(
        candidates,
        key=lambda row: (row["score"][1], row["high_speed_teacher_course_pass"], row["score"][2]),
    )
    qualified = best["score"][1] >= 5 and best["high_speed_teacher_course_pass"] >= 1
    report = {
        "schema": SCHEMA,
        "teacher_report_hash": teachers["report_hash"],
        "v202_report_hash": v202["report_hash"],
        "source_hashes": sources,
        "partition": "GENUINE_SPECIALIST_48D_DISTILLATION_CONSUMED_NOT_FRESH",
        "course_bank_hash": bank_hash,
        "teacher_trajectories": len(teachers["rows"]),
        "teacher_frames": len(states),
        "baseline_score": v202["development_score"],
        "candidates": candidates,
        "best_alpha": best["alpha"] if qualified else None,
        "status": "DEVELOPMENT_GENUINE_TEACHER_CANDIDATE_ONLY"
        if qualified
        else "REJECTED_GENUINE_TEACHER_DISTILL_GATE",
        "promotion_authorized": False,
        "video_authorized": False,
        "activation_ceiling": "SIM_ONLY",
    }
    report["report_hash"] = hash_json(report)
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if any(hash_bytes((root / name).read_bytes()) != digest for name, digest in sources.items()):
        raise ValueError("source drift during genuine teacher distillation")
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
        "old-zero-report",
        "kinematic-zero-dir",
        "v202-report",
        "teacher-dir",
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
        args.old_zero_report,
        args.kinematic_zero_dir,
        args.v202_report,
        args.teacher_dir,
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
