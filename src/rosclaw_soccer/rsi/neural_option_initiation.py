"""Finite-data initiation memory using ONLY own initial motor features."""

import math

import torch
from torch import nn


class NeuralOptionInitiation(nn.Module):
    normalized_templates: torch.Tensor
    labels: torch.Tensor
    scale: torch.Tensor
    radii: torch.Tensor

    def __init__(self, features: torch.Tensor, labels: torch.Tensor):
        super().__init__()
        if (
            features.device.type != "cpu"
            or features.ndim != 2
            or features.shape[1] != 288
            or not 2 <= len(features) <= 256
            or features.dtype != torch.float32
            or labels.device.type != "cpu"
            or labels.dtype != torch.int64
            or labels.shape != (len(features),)
            or ((labels < 0) | (labels > 3)).any()
            or not (labels == 0).any()
            or not (labels > 0).any()
            or not torch.isfinite(features).all()
        ):
            raise ValueError("bounded qualified own288 initiation features and0..3 labels required")
        scale = features.detach().double().std(dim=0, unbiased=False).clamp_min(0.001)
        normalized = features.detach().double() / scale
        distance = torch.cdist(
            normalized, normalized, compute_mode="donot_use_mm_for_euclid_dist"
        ) / math.sqrt(288)
        other = (
            distance.masked_fill(labels[:, None] == labels[None, :], float("inf")).min(dim=1).values
        )
        if (
            not torch.isfinite(other).all()
            or (other <= torch.finfo(torch.float64).eps ** 0.5).any()
        ):
            raise ValueError(
                "contradictory own-feature initiation labels cannot be routed by case IDs"
            )
        self.register_buffer("normalized_templates", normalized.detach().clone())
        self.register_buffer("labels", labels.detach().clone())
        self.register_buffer("scale", scale)
        self.register_buffer("radii", 0.45 * other)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if (
            features.ndim != 2
            or features.shape[1] != 288
            or not 1 <= len(features) <= 64
            or features.dtype != torch.float32
            or features.device != self.scale.device
            or not torch.isfinite(features).all()
            or self.scale.dtype != torch.float64
            or self.normalized_templates.dtype != torch.float64
            or self.radii.dtype != torch.float64
            or self.labels.dtype != torch.int64
            or self.labels.shape != (len(self.normalized_templates),)
            or ((self.labels < 0) | (self.labels > 3)).any()
            or any(
                not torch.isfinite(x).all()
                for x in (self.scale, self.normalized_templates, self.radii)
            )
            or not (self.scale >= 0.001).all()
            or not (self.radii > 0).all()
        ):
            raise ValueError("finite own initial query and unchanged initiation memory required")
        d = torch.cdist(
            features.detach().double() / self.scale,
            self.normalized_templates,
            compute_mode="donot_use_mm_for_euclid_dist",
        ) / math.sqrt(288)
        distance, index = d.min(dim=1)
        label = self.labels[index]
        return torch.where(distance <= self.radii[index], label, torch.zeros_like(label))
