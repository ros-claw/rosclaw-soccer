"""Experimental competence router: exact grown motor or its exact R2 ancestor.

Not a trained/checkpoint-admitted policy by itself. No case/source identifiers,
phase lookup or new torque authority. Both alternatives use the same private
own R2 recurrence. A router can be functionally harmless even while open, if
both motors propose exactly the same action in that state.
"""

import torch
from torch import nn

from rosclaw_soccer.rsi.paid_progressive_contact import ProgressiveContactResidual


class AncestralResidualRouter(nn.Module):
    bound_radians = 0.2

    def __init__(self, parent: ProgressiveContactResidual) -> None:
        super().__init__()
        if type(parent) is not ProgressiveContactResidual or parent.bound_radians != 0.2:
            raise ValueError("explicit qualified compound learned parent required")
        self.parent = parent.requires_grad_(False).eval()
        output = nn.Linear(64, 1)
        self.gate = nn.Sequential(nn.Linear(288, 64), nn.ELU(), output)
        nn.init.zeros_(output.weight)
        nn.init.constant_(output.bias, -4.0)

    def train(self, mode: bool = True) -> "AncestralResidualRouter":
        super().train(mode)
        self.parent.eval()
        return self

    def forward_with_gate(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            prior = self.parent(features)
        logits = self.gate(features)
        return torch.where(logits >= 0, torch.zeros_like(prior), prior), logits

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.forward_with_gate(features)[0]
