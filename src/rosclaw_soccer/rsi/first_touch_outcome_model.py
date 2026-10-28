"""Research-only outcome predictor; never a hardware or promotion authority."""

import torch
from torch import nn


class FirstTouchOutcomeModel(nn.Module):
    """Predict separate formal task-pass and physical safety probabilities."""

    def __init__(self):
        super().__init__()
        self.register_buffer("center", torch.tensor([2.5, 0.0, 0.0, 1.25, 0.775, 215.0, 0.0]))
        self.register_buffer("scale", torch.tensor([0.5, 0.3, 0.8, 0.75, 0.325, 35.0, 0.15]))
        self.mlp = nn.Sequential(
            nn.Linear(7, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 2)
        )

    def forward(self, inputs):
        if inputs.ndim != 2 or inputs.shape[1] != 7 or not torch.isfinite(inputs).all():
            raise ValueError("finite batched initial course and teacher parameters required")
        return self.mlp((inputs - self.center) / self.scale)
