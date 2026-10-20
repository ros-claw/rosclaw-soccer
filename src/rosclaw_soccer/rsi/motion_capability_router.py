"""G1 motion applicability learner; not generic Core promotion or consciousness."""

from typing import Any

import numpy as np


def motion_routing_features(current: np.ndarray) -> np.ndarray:
    if (
        type(current) is not np.ndarray
        or current.shape != (108,)
        or current.dtype != np.float32
        or not np.isfinite(current).all()
        or abs(float(np.linalg.norm(current[2:6])) - 1) > 1e-4
    ):
        raise ValueError("current owned108 body/ball observation required")
    upright = 1 - 2 * (float(current[3]) ** 2 + float(current[4]) ** 2)
    return np.concatenate(
        (np.asarray([current[1], upright], dtype=np.float32), current[70:76])
    ).astype(np.float32)


def create_motion_capability_router(*, experts: int = 5) -> Any:
    if type(experts) is not int or not 2 <= experts <= 16:
        raise ValueError("bounded frozen motion portfolio required")
    import torch
    from torch import nn

    class Router(nn.Module):
        observation_mean: torch.Tensor
        observation_std: torch.Tensor

        def __init__(self) -> None:
            super().__init__()
            self.expert_count = experts
            self.register_buffer("observation_mean", torch.zeros(8))
            self.register_buffer("observation_std", torch.ones(8))
            self.trunk = nn.Sequential(nn.Linear(8, 64), nn.ELU(), nn.Linear(64, 64), nn.ELU())
            self.owner = nn.Linear(64, experts)
            self.success = nn.Linear(64, experts)

        def forward(self, x: Any) -> tuple[Any, Any]:
            if (
                x.ndim != 2
                or x.shape[0] < 1
                or x.shape[1] != 8
                or x.dtype != torch.float32
                or not torch.isfinite(x).all()
                or not torch.isfinite(self.observation_mean).all()
                or not torch.isfinite(self.observation_std).all()
                or (self.observation_std < 0.01).any()
            ):
                raise ValueError("finite current8 and train-only normalizer required")
            h = self.trunk(((x - self.observation_mean) / self.observation_std).clamp(-10, 10))
            owner, success = self.owner(h), self.success(h)
            if not torch.isfinite(owner).all() or not torch.isfinite(success).all():
                raise ValueError("nonfinite motion applicability output")
            return owner, success

    return Router()
