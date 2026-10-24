"""Compact own-feature applicability memory, not a skill or safety certificate.

Future policy admission must bind these buffers to qualified actual data and
verify whole closed-loop behavior. A gate never enlarges motor authority.
"""

import math

import torch
from torch import nn


class FeatureSupportGate(nn.Module):
    """Smooth compact support with EXACT zero outside all measured rescue balls.

    Data defines anchors, scale and radii; no scenario IDs or private simulator
    inputs are accepted. The smooth union is1-product(1-kernel), avoiding hard
    nearest-expert switching. Known protected features outside every radius get
    bitwise zero. This alone proves nothing about unseen physical states.
    """

    scale: torch.Tensor
    normalized_anchors: torch.Tensor
    radii: torch.Tensor

    def __init__(self, anchors: torch.Tensor, scale: torch.Tensor, radii: torch.Tensor):
        super().__init__()
        if (
            anchors.device.type != "cpu"
            or anchors.dtype != torch.float32
            or anchors.ndim != 2
            or anchors.shape[1] != 288
            or not 1 <= len(anchors) <= 2048
            or scale.shape != (288,)
            or scale.dtype != torch.float64
            or scale.device.type != "cpu"
            or radii.shape != (len(anchors),)
            or radii.dtype != torch.float64
            or radii.device.type != "cpu"
            or not (scale >= 1e-3).all()
            or not (radii > 0).all()
            or any(not torch.isfinite(x).all() for x in (anchors, scale, radii))
        ):
            raise ValueError("exact finite own288 anchors/positive double scale and radii required")
        self.register_buffer("scale", scale.detach().clone())
        self.register_buffer("normalized_anchors", anchors.detach().double().clone() / self.scale)
        self.register_buffer("radii", radii.detach().clone())

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if (
            features.ndim not in (2, 3)
            or features.shape[-1] != 288
            or not 1 <= features.numel() // 288 <= 8192
            or features.dtype != torch.float32
            or features.device != self.scale.device
            or not torch.isfinite(features).all()
            or self.scale.dtype != torch.float64
            or self.radii.dtype != torch.float64
            or self.normalized_anchors.dtype != torch.float64
            or any(
                not torch.isfinite(t).all()
                for t in (self.scale, self.radii, self.normalized_anchors)
            )
            or not (self.scale >= 1e-3).all()
            or not (self.radii > 0).all()
        ):
            raise ValueError("finite own features and unchanged double support memory required")
        query = features.detach().reshape(-1, 288).double() / self.scale
        distance = torch.cdist(
            query, self.normalized_anchors, p=2, compute_mode="donot_use_mm_for_euclid_dist"
        ) / math.sqrt(288)
        inside = (1 - distance / self.radii).clamp(0, 1)
        kernel = inside.square() * (3 - 2 * inside)
        gate = 1 - (1 - kernel).prod(dim=1)
        if not torch.isfinite(gate).all() or ((gate < 0) | (gate > 1)).any():
            raise ValueError("bounded finite support activation required")
        return gate.float().reshape(*features.shape[:-1], 1)
