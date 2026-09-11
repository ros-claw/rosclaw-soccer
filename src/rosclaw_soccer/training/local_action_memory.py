"""Optional Torch phase-local action memory, with bounded plastic corrections.

This is a data-derived proposal model, not an actuator, admission policy, or
proof of generalization. Frozen exemplars preserve the original memory; only
bounded action corrections are trainable. Callers must enforce motion limits.
"""

from __future__ import annotations

import math
from typing import Any


def build_phase_action_memory(
    observations: Any,
    actions: Any,
    *,
    neighbors: int,
    phase_indices: tuple[int, int],
    phase_period: int,
    scale_floor: float = 0.01,
    maximum_correction: float = 0.05,
) -> Any:
    """Build a local, nonparametric policy with trainable bounded corrections.

    Banks have shape (phase, example, feature/action). Phase comes from the
    supplied sine/cosine observation channels, never a hidden execution clock.
    Unknown phases are rejected, not silently mapped to a successful example.
    """
    import torch

    if (
        not isinstance(observations, torch.Tensor)
        or not isinstance(actions, torch.Tensor)
        or observations.ndim != 3
        or actions.ndim != 3
        or observations.shape[:2] != actions.shape[:2]
        or observations.device != actions.device
        or any(t.dtype != torch.float32 or t.requires_grad for t in (observations, actions))
        or any(t.layout != torch.strided for t in (observations, actions))
        or any(t.numel() > 16_777_216 for t in (observations, actions))
        or not 1 <= observations.shape[0] <= 4096
        or not 2 <= observations.shape[1] <= 4096
        or not 2 <= observations.shape[2] <= 4096
        or not 1 <= actions.shape[2] <= 256
        or type(neighbors) is not int
        or not 1 <= neighbors <= observations.shape[1]
        or type(phase_period) is not int
        or not observations.shape[0] <= phase_period <= 10000
        or not isinstance(phase_indices, tuple)
        or len(phase_indices) != 2
        or any(type(i) is not int or not 0 <= i < observations.shape[2] for i in phase_indices)
        or phase_indices[0] == phase_indices[1]
        or any(
            type(v) not in (float, int) or not math.isfinite(v) or not 0 < v <= 1
            for v in (scale_floor, maximum_correction)
        )
        or not bool(torch.isfinite(observations).all())
        or not bool(torch.isfinite(actions).all())
    ):
        raise ValueError("bounded finite aligned phase memory required")

    def phases(value: Any) -> Any:
        sine, cosine = value[..., phase_indices[0]], value[..., phase_indices[1]]
        if not bool(((sine.square() + cosine.square() - 1).abs() <= 1e-4).all()):
            raise ValueError("normalized phase sine/cosine required")
        return (
            torch.round(
                torch.remainder(torch.atan2(sine, cosine), 2 * math.pi)
                * phase_period
                / (2 * math.pi)
            ).long()
            % phase_period
        )

    expected = torch.arange(len(observations), device=observations.device)[:, None]
    if not bool((phases(observations) == expected).all()):
        raise ValueError("memory examples must match their declared phase")
    scales = observations.std(dim=1).clamp_min(scale_floor)
    if not bool(torch.isfinite(scales).all()):
        raise ValueError("memory normalization overflow")

    class PhaseActionMemory(torch.nn.Module):
        centers: Any
        actions: Any
        scales: Any

        def __init__(self) -> None:
            super().__init__()
            self.register_buffer("centers", observations.clone())
            self.register_buffer("actions", actions.clone())
            self.register_buffer("scales", scales.clone())
            self.correction = torch.nn.Parameter(torch.zeros_like(actions))

        def forward(self, value: Any) -> Any:
            if (
                not isinstance(value, torch.Tensor)
                or value.ndim != 2
                or value.shape[1] != self.centers.shape[2]
                or not 1 <= len(value) <= 8192
                or value.dtype != torch.float32
                or value.device != self.centers.device
                or not bool(torch.isfinite(value).all())
            ):
                raise ValueError("aligned finite memory query required")
            phase = phases(value)
            if bool((phase >= len(self.centers)).any()):
                raise ValueError("query phase outside memory; explicit successor required")
            output = torch.empty(
                len(value), self.actions.shape[2], dtype=value.dtype, device=value.device
            )
            for index in phase.unique():
                selected = phase == index
                distance = (
                    ((self.centers[index][None] - value[selected, None]) / self.scales[index])
                    .square()
                    .mean(dim=2)
                )
                if not bool(torch.isfinite(distance).all()):
                    raise ValueError("memory distance overflow")
                nearest = torch.topk(distance, neighbors, largest=False, dim=1).indices
                bank = self.actions[index] + maximum_correction * self.correction[index].tanh()
                output[selected] = bank[nearest].mean(dim=1)
            if not bool(torch.isfinite(output).all()):
                raise ValueError("memory action overflow")
            return output

    return PhaseActionMemory()
