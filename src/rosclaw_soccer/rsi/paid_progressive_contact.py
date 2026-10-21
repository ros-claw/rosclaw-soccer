"""Experimental acquired-skill retention primitive, NOT a qualified policy loader.

An immutable learned parent is the fallback, rather than the original R2 alone.
Only a separate plastic head can learn. Features remain the same own-body288;
there is no new recurrent state, task-id router, physics or hardware authority.
The final residual relative to R2 retains the original .2rad bound. This module
alone is not a training result, retention certificate or deployment admission.
"""

import torch
from torch import nn

from rosclaw_soccer.rsi.paid_contact_residual import (
    ContactResidual,
    DecoupledContactResidual,
)


class ProgressiveContactResidual(nn.Module):
    """Frozen learned fallback plus a hard-gated bounded plastic correction.

    The new head proposes up to .4rad internally: switching from -.2 to +.2
    requires .4, but the *total applied* residual can never exceed .2. Rejected
    contexts return the parent's exact tensor values, including during training.
    That functional property is necessary, not sufficient, for physical retention:
    the gate must also be certified closed on acquired-success trajectory states.
    """

    bound_radians = 0.2

    def __init__(self, parent: ContactResidual, plastic: DecoupledContactResidual) -> None:
        super().__init__()
        if type(parent) not in (ContactResidual, DecoupledContactResidual):
            raise ValueError("one explicitly supported learned parent head required")
        if type(plastic) is not DecoupledContactResidual or parent is plastic:
            raise ValueError("independent decoupled plastic head required")
        parent_storage = {p.untyped_storage().data_ptr() for p in parent.parameters()}
        if any(p.untyped_storage().data_ptr() in parent_storage for p in plastic.parameters()):
            raise ValueError("parent and plastic tensors must not alias")
        if parent.bound_radians != 0.2 or plastic.bound_radians != 0.2:
            raise ValueError("unchanged .2rad component authority required")
        self.parent = parent.requires_grad_(False).eval()
        self.plastic = plastic

    def train(self, mode: bool = True) -> "ProgressiveContactResidual":
        super().train(mode)
        # Recursive .train() must not undo the frozen parent's reject semantics.
        self.parent.eval()
        return self

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # Parent gradients are never useful for plastic training, even if the
        # caller supplies differentiable features. No detach of plastic inputs.
        with torch.no_grad():
            prior = self.parent(features)
        correction, logits = self.plastic.forward_with_gate(features)
        candidate = torch.clamp(prior + 2.0 * correction, -self.bound_radians, self.bound_radians)
        # Do not express rejection as adding a masked zero: exact learned-parent
        # fallback is part of the contract, not a tolerance-based similarity.
        return torch.where(logits >= 0, candidate, prior)

    def plastic_parameters(self) -> tuple[nn.Parameter, ...]:
        """Only these tensors may enter a future adaptation optimizer."""
        return tuple(p for p in self.plastic.parameters() if p.requires_grad)
