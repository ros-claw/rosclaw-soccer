"""Exhaustive checkpoint selection with exact MJX failure-state exams.

The ordinary Brax evaluation averages over full routes and can hide the local
states that caused a recovery policy to regress.  This SIM_ONLY orchestrator
replays every persisted checkpoint with the same failure bank and the same
random seed, then composes those reports with the normal-route selector.  It
never grants deployment or promotion authority.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from rosclaw_soccer.training.opentrack_recovery_mjx_failure_exam import (
    RecoveryMJXFailureStateExamConfig,
    run_opentrack_recovery_mjx_failure_state_exam,
)
from rosclaw_soccer.training.recovery_mjx import (
    RecoveryMJXFailureConstraintConfig,
    select_recovery_mjx_failure_constrained_generation,
    validate_recovery_mjx_teacher_residual_report,
)


def run_opentrack_recovery_mjx_constrained_checkpoint_selection(
    *,
    opentrack_root: Path,
    teacher_checkpoint_path: Path,
    teacher_config_path: Path,
    parent_actor_checkpoint_path: Path,
    snapshot_manifest_path: Path,
    failure_state_manifest_path: Path,
    training_report_path: Path,
    generation_selection_path: Path,
    output_dir: Path,
    source_checkout_path: Path,
    exam_config: RecoveryMJXFailureStateExamConfig | None = None,
    constraint_config: RecoveryMJXFailureConstraintConfig | None = None,
) -> dict[str, Any]:
    """Run one comparable local exam per persisted PPO checkpoint."""

    training_path = training_report_path.expanduser().resolve()
    destination = output_dir.expanduser().resolve()
    checkout = source_checkout_path.expanduser().resolve()
    training = validate_recovery_mjx_teacher_residual_report(training_path)
    checkpoint_root = training_path.parent / "checkpoints"
    checkpoint_steps = sorted(
        {
            int(str(row["path"]).split("/", maxsplit=1)[0])
            for row in training["candidate_checkpoint_files"]
            if isinstance(row, dict)
            and isinstance(row.get("path"), str)
            and str(row["path"]).split("/", maxsplit=1)[0].isdigit()
        }
    )
    checkpoint_paths = [checkpoint_root / f"{step:012d}" for step in checkpoint_steps]
    if (
        destination.exists()
        or destination == checkout
        or checkout in destination.parents
        or not checkpoint_steps
        or not all(path.is_dir() for path in checkpoint_paths)
    ):
        raise ValueError("constrained checkpoint selection paths are invalid")
    active_exam = exam_config or RecoveryMJXFailureStateExamConfig()
    active_constraint = constraint_config or RecoveryMJXFailureConstraintConfig()
    destination.mkdir(parents=True)
    exam_paths: list[Path] = []
    for step, candidate_checkpoint in zip(checkpoint_steps, checkpoint_paths, strict=True):
        exam_path = destination / f"failure-exam-step-{step:012d}.json"
        run_opentrack_recovery_mjx_failure_state_exam(
            opentrack_root=opentrack_root,
            teacher_checkpoint_path=teacher_checkpoint_path,
            teacher_config_path=teacher_config_path,
            parent_actor_checkpoint_path=parent_actor_checkpoint_path,
            candidate_actor_checkpoint_path=candidate_checkpoint,
            snapshot_manifest_path=snapshot_manifest_path,
            failure_state_manifest_path=failure_state_manifest_path,
            output_path=exam_path,
            source_checkout_path=checkout,
            config=active_exam,
        )
        exam_paths.append(exam_path)
    return select_recovery_mjx_failure_constrained_generation(
        training_report_path=training_path,
        generation_selection_path=generation_selection_path,
        failure_state_exam_paths=tuple(exam_paths),
        output_path=destination / "failure-constrained-selection.json",
        config=active_constraint,
    )


def _main() -> None:
    parser = argparse.ArgumentParser(
        description="Run exact failure-state exams for every MJX PPO checkpoint"
    )
    parser.add_argument("--opentrack-root", required=True, type=Path)
    parser.add_argument("--teacher-checkpoint", required=True, type=Path)
    parser.add_argument("--teacher-config", required=True, type=Path)
    parser.add_argument("--parent-actor-checkpoint", required=True, type=Path)
    parser.add_argument("--snapshot-manifest", required=True, type=Path)
    parser.add_argument("--failure-state-manifest", required=True, type=Path)
    parser.add_argument("--training-report", required=True, type=Path)
    parser.add_argument("--generation-selection", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--source-checkout", required=True, type=Path)
    parser.add_argument("--num-environments", default=384, type=int)
    parser.add_argument("--horizon-steps", default=400, type=int)
    parser.add_argument("--seed", default=5492, type=int)
    parser.add_argument("--dual-learning-rate", default=0.5, type=float)
    args = parser.parse_args()
    result = run_opentrack_recovery_mjx_constrained_checkpoint_selection(
        opentrack_root=args.opentrack_root,
        teacher_checkpoint_path=args.teacher_checkpoint,
        teacher_config_path=args.teacher_config,
        parent_actor_checkpoint_path=args.parent_actor_checkpoint,
        snapshot_manifest_path=args.snapshot_manifest,
        failure_state_manifest_path=args.failure_state_manifest,
        training_report_path=args.training_report,
        generation_selection_path=args.generation_selection,
        output_dir=args.output_dir,
        source_checkout_path=args.source_checkout,
        exam_config=RecoveryMJXFailureStateExamConfig(
            num_environments=args.num_environments,
            horizon_steps=args.horizon_steps,
            random_seed=args.seed,
        ),
        constraint_config=RecoveryMJXFailureConstraintConfig(
            dual_learning_rate=args.dual_learning_rate
        ),
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    _main()


__all__ = ["run_opentrack_recovery_mjx_constrained_checkpoint_selection"]
