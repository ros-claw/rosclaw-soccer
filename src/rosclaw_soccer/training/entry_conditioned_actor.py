"""Simulation-only entry-memory adapter; no collector or policy activation.

The last six observations are episode-entry ball-relative XYZ and linear XYZ
velocity in the motor observation frame, held constant by the collector. Never
fill them with later outcomes. Calibration must use development entries only.
The original 136-feature actor is frozen; zero heads preserve its arithmetic.
This is a residual policy experiment, not end-to-end torque control.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from rosclaw_soccer.training.goal_reference_actor import build_goal_reference_actor_critic


def build_entry_conditioned_actor_critic(
    parent_state: Mapping[str, Any], entry_center: Any, entry_scale: Any
) -> Any:
    """Create a bounded trainable adapter around a numeric goal/reference seed."""
    import torch

    def numeric(value: Any) -> bool:
        return (
            isinstance(value, torch.Tensor)
            and value.device.type == "cpu"
            and value.dtype == torch.float32
            and value.layout == torch.strided
            and bool(torch.isfinite(value).all())
        )

    if (
        not isinstance(parent_state, Mapping)
        or not parent_state
        or not all(numeric(v) for v in parent_state.values())
        or not numeric(entry_center)
        or not numeric(entry_scale)
        or entry_center.shape != (6,)
        or entry_scale.shape != (6,)
        or bool((entry_center.abs() > 10).any())
        or bool(((entry_scale < 0.001) | (entry_scale > 10)).any())
    ):
        raise ValueError("finite numeric parent and bounded six-feature calibration required")
    if "logstd" not in parent_state or parent_state["logstd"].shape != (30,):
        raise ValueError("30-action goal/reference parent required")
    seed = {
        k: v.detach().clone()
        for k, v in parent_state.items()
        if k.startswith(("actor.", "critic."))
    }
    seed["logstd"] = parent_state["logstd"][:29].detach().clone()
    parent = build_goal_reference_actor_critic(seed)
    parent.load_state_dict(parent_state, strict=True)
    parent.requires_grad_(False)

    class EntryConditionedActorCritic(torch.nn.Module):
        entry_center: torch.Tensor
        entry_scale: torch.Tensor

        def __init__(self) -> None:
            super().__init__()
            self.parent = parent
            self.logstd = torch.nn.Parameter(parent.logstd.detach().clone())
            self.register_buffer("entry_center", entry_center.detach().clone())
            self.register_buffer("entry_scale", entry_scale.detach().clone())

            def head(outputs: int) -> Any:
                last = torch.nn.Linear(32, outputs)
                torch.nn.init.zeros_(last.weight)
                torch.nn.init.zeros_(last.bias)
                return torch.nn.Sequential(torch.nn.Linear(6, 32), torch.nn.Tanh(), last)

            self.entry_actor = head(30)
            self.entry_critic = head(1)

        def forward(self, observation: Any) -> tuple[Any, Any]:
            if (
                not isinstance(observation, torch.Tensor)
                or observation.ndim != 2
                or observation.shape[1] != 142
                or not 1 <= observation.shape[0] <= 65536
                or observation.layout != torch.strided
                or observation.dtype != self.logstd.dtype
                or observation.device != self.logstd.device
                or not bool(torch.isfinite(observation).all())
                or bool((observation.abs() > 10).any())
            ):
                raise ValueError("finite bounded (N, 142) entry-memory observations required")
            # Loading a checkpoint must not bypass calibration validation.
            if (
                not bool(torch.isfinite(self.entry_center).all())
                or not bool(torch.isfinite(self.entry_scale).all())
                or bool((self.entry_center.abs() > 10).any())
                or bool(((self.entry_scale < 0.001) | (self.entry_scale > 10)).any())
            ):
                raise ValueError("invalid checkpoint entry calibration")
            context = ((observation[:, 136:] - self.entry_center) / self.entry_scale).clamp(-5, 5)
            mean, value = self.parent(observation[:, :136].contiguous())
            mean = mean + 0.5 * torch.tanh(self.entry_actor(context))
            value = value + self.entry_critic(context).squeeze(-1)
            if not bool(torch.isfinite(mean).all() and torch.isfinite(value).all()):
                raise FloatingPointError("nonfinite entry-memory actor or critic output")
            return mean, value

    return EntryConditionedActorCritic()
