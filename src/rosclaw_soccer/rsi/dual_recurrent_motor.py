"""Research-only stable-memory + plastic end-to-end recurrent motor network.

Both recurrent branches consume the SAME actual observation stream, but own
different h/c. Frozen skill memory never consumes the learner's hidden state.
The packed256 context is a new contract, NOT an old128 teacher context.
Untrained construction/mean bounds do not prove old physical skill retention.
"""

import copy
from typing import Any

import torch
from torch import nn

from rosclaw_soccer.rsi.paid_full_entry_student import FrozenFullEntryActor


class DualRecurrentMotor(nn.Module):
    hidden_width = 256
    simulation_only = True
    policy_domain = "research.dual_recurrent_motor.v1"

    def __init__(self, parent: FrozenFullEntryActor) -> None:
        super().__init__()
        if type(parent) is not FrozenFullEntryActor:
            raise ValueError("exact source-qualified frozen full-entry motor required")
        self.parent = parent.requires_grad_(False).eval()
        self.plastic = copy.deepcopy(parent.base).requires_grad_(True)
        self.plastic.eval()
        storage = {p.untyped_storage().data_ptr() for p in self.parent.parameters()}
        if any(p.untyped_storage().data_ptr() in storage for p in self.plastic.parameters()):
            raise ValueError("plastic weights cannot alias frozen motor memory")

    def train(self, mode: bool = True) -> "DualRecurrentMotor":
        super().train(mode)
        self.parent.eval()
        return self

    def initial_state(self, *, batch: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
        if type(batch) is not int or not 1 <= batch <= 64:
            raise ValueError("bounded explicit private player batch required")
        p = next(self.plastic.parameters())
        h = torch.zeros(2, batch, 256, dtype=p.dtype, device=p.device)
        return h, torch.zeros_like(h)

    def forward_with_evidence(
        self, obs: torch.Tensor, state: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor], dict[str, Any]]:
        if (
            obs.ndim != 3
            or obs.shape[-1] != 160
            or not 1 <= obs.shape[0] <= 512
            or not 1 <= obs.shape[1] <= 64
            or obs.dtype != torch.float32
            or len(state) != 2
            or not torch.isfinite(obs).all()
            or any(
                t.shape != (2, obs.shape[1], 256)
                or t.dtype != obs.dtype
                or t.device != obs.device
                or not torch.isfinite(t).all()
                for t in state
            )
        ):
            raise ValueError("new private dual256 history and complete actual160 stream required")
        reference_state = tuple(t[..., :128].contiguous() for t in state)
        plastic_state = tuple(t[..., 128:].contiguous() for t in state)
        scale = self.parent.get_buffer("action_scale")
        with torch.no_grad():
            nominal, reference_after, hidden = self.parent.base.forward_with_features(
                obs, reference_state
            )
            features = torch.cat(
                ((obs - self.parent.base.mean) / self.parent.base.denominator, hidden), dim=-1
            )
            prior = self.parent.residual(features.reshape(-1, 288)).reshape(*nominal.shape)
            if not torch.isfinite(prior).all() or (prior.abs() > 0.2).any():
                raise ValueError("frozen source motor authority changed")
            reference_mean = nominal + prior / scale
        learned, plastic_after, learned_hidden = self.plastic.forward_with_features(
            obs, plastic_state
        )
        if (
            learned.shape != nominal.shape
            or learned.dtype != obs.dtype
            or not torch.isfinite(learned).all()
            or len(plastic_after) != 2
            or any(
                t.shape != (2, obs.shape[1], 128) or not torch.isfinite(t).all()
                for t in plastic_after
            )
        ):
            raise ValueError("finite private plastic motor mean and actual next memory required")
        plastic_delta = (learned - nominal) * scale
        final_delta = torch.clamp(prior + plastic_delta, -0.2, 0.2)
        mean = nominal + final_delta / scale
        after = (
            torch.cat((reference_after[0], plastic_after[0]), dim=-1),
            torch.cat((reference_after[1], plastic_after[1]), dim=-1),
        )
        return (
            mean,
            after,
            dict(
                reference_nominal=nominal,
                reference_mean=reference_mean,
                critic_features=torch.cat(
                    ((obs - self.parent.base.mean) / self.parent.base.denominator, learned_hidden),
                    dim=-1,
                ),
                final_mean_delta_radians=final_delta,
                plastic_delta_before_limit=plastic_delta,
            ),
        )

    def forward(
        self, obs: torch.Tensor, state: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        mean, after, _ = self.forward_with_evidence(obs, state)
        return mean, after

    def plastic_parameters(self) -> tuple[nn.Parameter, ...]:
        return tuple(p for p in self.plastic.parameters() if p.requires_grad)
