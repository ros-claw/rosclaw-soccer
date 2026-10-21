"""TRAIN-derived compact support, not a case-ID router or a physical certificate.

Each teacher centre owns a continuous cone with radius .45 times its nearest
protected-state distance. Every protected anchor is outside every cone. This
constructively separates exact ancestral retention from head optimization;
it promises nothing about unobserved states or closed-loop trajectory drift.
"""

import torch
from torch import nn


def check_features(x: torch.Tensor) -> None:
    if (
        x.ndim != 2
        or x.shape[1] != 288
        or not 1 <= len(x) <= 50000
        or x.dtype != torch.float32
        or not torch.isfinite(x).all()
    ):
        raise ValueError("bounded finite float32 own288 features required")


def nearest_protected_radii(positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
    """Direct float64 subtraction avoids norm-square cancellation near anchors."""
    check_features(positive)
    check_features(negative)
    if positive.device != negative.device:
        raise ValueError("same geometry device required")
    result = []
    with torch.no_grad():
        for start in range(0, len(positive), 16):
            x = positive[start : start + 16].double()
            minimum = torch.full((len(x),), float("inf"), dtype=torch.float64, device=x.device)
            for j in range(0, len(negative), 512):
                d = (x[:, None] - negative[j : j + 512].double()[None]).square().sum(-1)
                minimum = torch.minimum(minimum, d.min(dim=1).values)
            if not (minimum > 0).all():
                raise ValueError("teacher/protected observation collision: revise observability")
            result.append(0.45 * minimum.sqrt())
    return torch.cat(result)


class CompactTeacherSupport(nn.Module):
    centres: torch.Tensor
    radii: torch.Tensor

    def __init__(self, centres: torch.Tensor, radii: torch.Tensor) -> None:
        super().__init__()
        check_features(centres)
        if (
            radii.shape != (len(centres),)
            or radii.dtype != torch.float64
            or radii.device != centres.device
            or not torch.isfinite(radii).all()
            or not (radii > 0).all()
        ):
            raise ValueError("one finite strictly positive float64 radius per teacher required")
        self.register_buffer("centres", centres.detach().clone())
        self.register_buffer("radii", radii.detach().clone())

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        check_features(features)
        if features.device != self.centres.device:
            raise ValueError("support/query device differs")
        # Geometry is fixed memory; only the motor head is plastic. Bounded
        # chunks avoid an unbounded [batch, all_centres, 288] temporary.
        rows = []
        with torch.no_grad():
            for start in range(0, len(features), 16):
                x = features[start : start + 16].double()
                gain = torch.zeros(len(x), dtype=torch.float64, device=x.device)
                for j in range(0, len(self.centres), 256):
                    d = (x[:, None] - self.centres[j : j + 256].double()[None]).square().sum(-1)
                    cone = (1 - d.sqrt() / self.radii[j : j + 256]).clamp(0, 1)
                    gain = torch.maximum(gain, cone.max(dim=1).values)
                rows.append(gain.float().reshape(-1, 1))
        return torch.cat(rows)


class CompactSupportedHead(nn.Module):
    bound_radians = 0.2

    def __init__(self, support: CompactTeacherSupport) -> None:
        super().__init__()
        if type(support) is not CompactTeacherSupport:
            raise ValueError("explicit fixed TRAIN support required")
        self.support = support
        self.net = nn.Sequential(
            nn.Linear(288, 128), nn.ELU(), nn.Linear(128, 64), nn.ELU(), nn.Linear(64, 29)
        )
        output = self.net[-1]
        if not isinstance(output, nn.Linear):
            raise ValueError("explicit linear29 motor output required")
        nn.init.zeros_(output.weight)
        nn.init.zeros_(output.bias)

    def correction(self, features: torch.Tensor) -> torch.Tensor:
        check_features(features)
        return 0.4 * torch.tanh(self.net(features))

    def combine(
        self, features: torch.Tensor, prior: torch.Tensor, gain: torch.Tensor | None = None
    ) -> torch.Tensor:
        check_features(features)
        if (
            prior.shape != (len(features), 29)
            or prior.dtype != torch.float32
            or prior.device != features.device
            or not torch.isfinite(prior).all()
            or (prior.abs() > 0.2).any()
        ):
            raise ValueError("finite frozen-parent batch29 within original authority required")
        # Explicit cached gains are for audited TRAIN centre/anchor minibatches
        # only. Runtime callers must evaluate support from their own features.
        if gain is None:
            gain = self.support(features)
        if (
            gain.shape != (len(features), 1)
            or gain.dtype != torch.float32
            or gain.device != features.device
            or not torch.isfinite(gain).all()
            or (gain < 0).any()
            or (gain > 1).any()
        ):
            raise ValueError("finite [0,1] support gain required")
        candidate = (prior + gain * self.correction(features)).clamp(-0.2, 0.2)
        return torch.where(gain > 0, candidate, prior)
