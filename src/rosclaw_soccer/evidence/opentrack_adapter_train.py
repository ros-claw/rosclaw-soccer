"""Bounded OpenTrack adapter-training launcher for ROSClaw evidence runs."""

from __future__ import annotations

import argparse
import importlib
import math
import os
import re
from dataclasses import dataclass


@dataclass(frozen=True)
class AdapterStepBudget:
    """Exact world-step quantum imposed by the upstream PPO batch shape."""

    sealed_maximum_world_steps: int
    step_quantum: int
    expected_world_steps: int

    @property
    def aligned(self) -> bool:
        return self.expected_world_steps == self.sealed_maximum_world_steps


def calculate_adapter_step_budget(
    *,
    sealed_maximum_world_steps: int,
    batch_size: int,
    unroll_length: int,
    num_minibatches: int,
    action_repeat: int,
    num_evals: int,
    num_resets_per_eval: int,
) -> AdapterStepBudget:
    """Mirror OpenTrack's ceil-based MBPPO accounting before launching GPUs."""

    values = (
        sealed_maximum_world_steps,
        batch_size,
        unroll_length,
        num_minibatches,
        action_repeat,
    )
    if min(values) <= 0 or min(num_evals, num_resets_per_eval) < 0:
        raise ValueError("OpenTrack step-budget dimensions are invalid")
    epochs = max(num_evals - 1, 1)
    resets = max(num_resets_per_eval, 1)
    step_quantum = batch_size * unroll_length * num_minibatches * action_repeat
    epoch_quantum = epochs * step_quantum * resets
    expected = math.ceil(sealed_maximum_world_steps / epoch_quantum) * epoch_quantum
    return AdapterStepBudget(
        sealed_maximum_world_steps=sealed_maximum_world_steps,
        step_quantum=epoch_quantum,
        expected_world_steps=expected,
    )


@dataclass(frozen=True)
class StableAdapterTrainingOverrides:
    """Loss overrides that keep residual exploration subordinate to tracking."""

    supervised_loss_weight: float
    entropy_cost: float
    policy_learning_rate: float
    world_model_learning_rate: float
    rehearsal_fraction: float
    acquisition_fraction: float
    maximum_world_steps: int
    activation_ceiling: str = "SIM_ONLY"

    def __post_init__(self) -> None:
        for label, value in (
            ("supervised_loss_weight", self.supervised_loss_weight),
            ("entropy_cost", self.entropy_cost),
            ("policy_learning_rate", self.policy_learning_rate),
            ("world_model_learning_rate", self.world_model_learning_rate),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{label} must be finite and positive")
        if (
            not math.isclose(
                self.rehearsal_fraction + self.acquisition_fraction, 1.0, abs_tol=1e-12
            )
            or min(self.rehearsal_fraction, self.acquisition_fraction) <= 0.0
        ):
            raise ValueError("rehearsal and acquisition fractions must be positive and sum to one")
        if self.maximum_world_steps <= 0:
            raise ValueError("maximum_world_steps must be positive")
        if self.activation_ceiling != "SIM_ONLY":
            raise ValueError("OpenTrack adapter training is SIM_ONLY")


def run_stable_adapter_training(
    *,
    task: str,
    parent_experiment: str,
    experiment_name: str,
    trajectory_names: tuple[str, ...],
    overrides: StableAdapterTrainingOverrides,
    num_envs: int = 4096,
    batch_size: int = 512,
    seed: int = 0,
) -> None:
    """Delegate to OpenTrack after applying audited stability loss weights."""

    identifier = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
    for label, value in (
        ("task", task),
        ("parent_experiment", parent_experiment),
        ("experiment_name", experiment_name),
    ):
        if not identifier.fullmatch(value):
            raise ValueError(f"{label} is not normalized")
    if len(trajectory_names) < 2 or len(set(trajectory_names)) != len(trajectory_names):
        raise ValueError("stable adapter training requires unique rehearsal/acquisition motions")
    if min(num_envs, batch_size) <= 0 or num_envs % batch_size:
        raise ValueError("num_envs must be a positive multiple of batch_size")

    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    tmj = importlib.import_module("track_mj")
    train_module = importlib.import_module("track_mj.learning.train.train_adapter")
    task_cfg = tmj.registry.get(task, "tracking_adapter_config")
    task_cfg.policy_config.entropy_cost = overrides.entropy_cost
    task_cfg.mbppo_policy_config.supervised_loss_weight = overrides.supervised_loss_weight
    budget = calculate_adapter_step_budget(
        sealed_maximum_world_steps=overrides.maximum_world_steps,
        batch_size=batch_size,
        unroll_length=int(task_cfg.policy_config.unroll_length),
        num_minibatches=int(task_cfg.policy_config.num_minibatches),
        action_repeat=int(task_cfg.policy_config.action_repeat),
        num_evals=int(task_cfg.policy_config.num_evals),
        num_resets_per_eval=int(task_cfg.policy_config.num_resets_per_eval),
    )
    if not budget.aligned:
        raise ValueError(
            "sealed maximum_world_steps is not aligned to the OpenTrack PPO quantum: "
            f"requested={budget.sealed_maximum_world_steps}, "
            f"quantum={budget.step_quantum}, expected={budget.expected_world_steps}"
        )
    args = train_module.Args(
        task=task,
        load_exp_name=parent_experiment,
        exp_name=experiment_name,
        seed=seed,
        convert_onnx=False,
        num_timesteps=overrides.maximum_world_steps,
        num_envs=num_envs,
        batch_size=batch_size,
        use_adapter=True,
        use_world_model=True,
        policy_lr=overrides.policy_learning_rate,
        world_model_lr=overrides.world_model_learning_rate,
        trajectory_name=",".join(trajectory_names),
    )
    train_module.train(args)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a stability-preserving OpenTrack adapter")
    parser.add_argument("--task", default="G1TrackingGeneralDR")
    parser.add_argument("--parent-experiment", required=True)
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--trajectory-name", required=True, action="append")
    parser.add_argument("--maximum-world-steps", required=True, type=int)
    parser.add_argument("--supervised-loss-weight", required=True, type=float)
    parser.add_argument("--entropy-cost", required=True, type=float)
    parser.add_argument("--policy-learning-rate", required=True, type=float)
    parser.add_argument("--world-model-learning-rate", required=True, type=float)
    parser.add_argument("--rehearsal-fraction", default=0.6, type=float)
    parser.add_argument("--acquisition-fraction", default=0.4, type=float)
    parser.add_argument("--num-envs", default=4096, type=int)
    parser.add_argument("--batch-size", default=512, type=int)
    parser.add_argument("--seed", default=0, type=int)
    args = parser.parse_args()
    overrides = StableAdapterTrainingOverrides(
        supervised_loss_weight=args.supervised_loss_weight,
        entropy_cost=args.entropy_cost,
        policy_learning_rate=args.policy_learning_rate,
        world_model_learning_rate=args.world_model_learning_rate,
        rehearsal_fraction=args.rehearsal_fraction,
        acquisition_fraction=args.acquisition_fraction,
        maximum_world_steps=args.maximum_world_steps,
    )
    run_stable_adapter_training(
        task=args.task,
        parent_experiment=args.parent_experiment,
        experiment_name=args.experiment_name,
        trajectory_names=tuple(args.trajectory_name),
        overrides=overrides,
        num_envs=args.num_envs,
        batch_size=args.batch_size,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()


__all__ = [
    "AdapterStepBudget",
    "StableAdapterTrainingOverrides",
    "calculate_adapter_step_budget",
    "run_stable_adapter_training",
]
