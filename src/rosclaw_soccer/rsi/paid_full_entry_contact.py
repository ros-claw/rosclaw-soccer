"""Untrained first-entry plastic motor primitive over a frozen competence router.

Unlike two-way routing this head can learn genuinely new teacher actions.
No checkpoint admission, runtime integration, physics qualification or promotion
is implied by constructing it. Same own288 inputs and final .2rad authority.
"""

import torch
from torch import nn

from rosclaw_soccer.rsi.paid_ancestor_router import AncestralResidualRouter
from rosclaw_soccer.rsi.paid_contact_residual import DecoupledContactResidual


class FullEntryContactResidual(nn.Module):
    bound_radians = 0.2

    def __init__(self, parent: AncestralResidualRouter, plastic: DecoupledContactResidual) -> None:
        super().__init__()
        if (
            type(parent) is not AncestralResidualRouter
            or type(plastic) is not DecoupledContactResidual
        ):
            raise ValueError("explicit frozen router and independent fresh contact head required")
        if parent.bound_radians != 0.2 or plastic.bound_radians != 0.2:
            raise ValueError("unchanged R2-relative .2rad final authority required")
        storage = {p.untyped_storage().data_ptr() for p in parent.parameters()}
        if any(p.untyped_storage().data_ptr() in storage for p in plastic.parameters()):
            raise ValueError("new plastic memory cannot alias any ancestral tensor")
        self.parent = parent.requires_grad_(False).eval()
        self.plastic = plastic

    def train(self, mode: bool = True) -> "FullEntryContactResidual":
        super().train(mode)
        self.parent.eval()
        return self

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            prior = self.parent(features)
        delta, logits = self.plastic.forward_with_gate(features)
        candidate = torch.clamp(prior + 2.0 * delta, -0.2, 0.2)
        return torch.where(logits >= 0, candidate, prior)

    def plastic_parameters(self) -> tuple[nn.Parameter, ...]:
        return tuple(p for p in self.plastic.parameters() if p.requires_grad)
