"""Causal task-conditioned continuation of an existing receiving command NN.

A zero-initialized task projection is added to the first hidden preactivation.
The old112 normalization and six trained tensors are preserved; unlike a new
random actor, the initial policy is exactly the admitted parent function.
Still original bounded velocity commands, not joint torque or physical proof.
"""

import copy

import numpy as np
import torch
from torch import nn

from rosclaw_soccer.rsi.receiving_command_actor import ReceivingCommandActor


def receiving_task_context6(
    root_pose: np.ndarray,
    capture_goal_world: np.ndarray,
    pass_direction_world: np.ndarray,
) -> np.ndarray:
    """Communicated goal displacement3 and direction3 in the CURRENT root frame."""
    if (
        root_pose.shape != (7,)
        or capture_goal_world.shape != (3,)
        or pass_direction_world.shape != (2,)
        or any(
            x.dtype != np.float64 or not np.isfinite(x).all()
            for x in (root_pose, capture_goal_world, pass_direction_world)
        )
        or abs(np.linalg.norm(root_pose[3:7]) - 1) > 1e-5
        or abs(np.linalg.norm(pass_direction_world) - 1) > 1e-10
    ):
        raise ValueError(
            "actual finite root pose and communicated goal/unit pass direction required"
        )
    w, x, y, z = root_pose[3:7]
    rotation = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
    result: np.ndarray = np.r_[
        rotation.T @ (capture_goal_world - root_pose[:3]),
        rotation.T @ np.r_[pass_direction_world, 0.0],
    ].astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError("finite own-frame task conditions required, never future outcome")
    return result


class ReceivingTaskCommandActor(nn.Module):
    task_mean: torch.Tensor
    task_scale: torch.Tensor
    simulation_only = True
    policy_domain = "research.receiver_neural_task_command.v2"

    def __init__(
        self, parent: ReceivingCommandActor, task_mean: torch.Tensor, task_scale: torch.Tensor
    ) -> None:
        super().__init__()
        if (
            type(parent) is not ReceivingCommandActor
            or any(
                p.dtype != torch.float32 or p.device.type != "cpu" or not torch.isfinite(p).all()
                for p in parent.parameters()
            )
            or any(
                x.shape != (6,)
                or x.dtype != torch.float32
                or x.device.type != "cpu"
                or x.requires_grad
                or not torch.isfinite(x).all()
                for x in (task_mean, task_scale)
            )
            or (task_scale < 0.1).any()
        ):
            raise ValueError(
                "exact finite CPU parent and fixed TRAIN-only task normalization required"
            )
        self.prior = copy.deepcopy(parent).requires_grad_(True)
        self.register_buffer("task_mean", task_mean.clone())
        self.register_buffer("task_scale", task_scale.clone())
        self.task_projection = nn.Linear(6, 128, bias=False, dtype=torch.float32)
        nn.init.zeros_(self.task_projection.weight)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        if (
            observations.ndim != 2
            or observations.shape[1] != 118
            or not 1 <= len(observations) <= 8192
            or observations.dtype != torch.float32
            or observations.device != self.task_mean.device
            or not torch.isfinite(observations).all()
        ):
            raise ValueError("finite own112 plus communicated BODYFRAME task6 required")
        original = (
            observations[:, :112] - self.prior.observation_mean
        ) / self.prior.observation_scale
        task = (observations[:, 112:] - self.task_mean) / self.task_scale
        hidden = self.prior.net[0](original) + self.task_projection(task)
        logits = self.prior.net[1:](hidden)
        commands = self.prior.command_center + self.prior.command_radius * torch.tanh(logits)
        if not torch.isfinite(commands).all():
            raise ValueError("nonfinite task-conditioned command; never repair or execute")
        return commands
