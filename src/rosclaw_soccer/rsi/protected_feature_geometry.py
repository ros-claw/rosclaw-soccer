"""Data-only support separability; no action, physics or learned policy claims."""

import math

import torch


def protected_feature_geometry(
    protected: torch.Tensor, corrective: torch.Tensor
) -> dict[str, torch.Tensor]:
    """Nearest protected-feature margins for a prospective local rescue memory.

    Fit scale ONLY on protected own features. A radius below half the nearest
    protected distance separates a corrective witness from this finite set;
    it does not prove unseen-state coverage or closed-loop intervention reach.
    No state/task IDs, teacher recurrent states or exam labels enter distances.
    """
    for x, maximum in ((protected, 50000), (corrective, 2048)):
        if (
            x.device.type != "cpu"
            or x.dtype != torch.float32
            or x.ndim != 2
            or x.shape[1] != 288
            or not 1 <= len(x) <= maximum
            or not torch.isfinite(x).all()
        ):
            raise ValueError("bounded finite own288 CPU feature matrices required")
    with torch.no_grad():
        scale = protected.double().std(dim=0, unbiased=False).clamp_min(1e-3)
        anchor = corrective.double() / scale
        closest = torch.full((len(anchor),), float("inf"), dtype=torch.float64)
        closest_index = torch.zeros(len(anchor), dtype=torch.int64)
        for start in range(0, len(protected), 512):
            block = protected[start : start + 512].double() / scale
            distances = torch.cdist(anchor, block, p=2, compute_mode="donot_use_mm_for_euclid_dist")
            value, index = distances.min(dim=1)
            use = value < closest
            closest = torch.where(use, value, closest)
            closest_index = torch.where(use, index + start, closest_index)
        # No norm-squared/MM cancellation: exact duplicate witnesses stay zero.
        difference = (corrective.double() - protected[closest_index].double()) / scale
        distance = difference.square().mean(dim=1).sqrt()
        radius = distance * 0.45
        if not torch.isfinite(distance).all() or not torch.isfinite(scale).all():
            raise ValueError("finite measured feature separability required")
        return dict(
            scale=scale,
            nearest_protected_index=closest_index,
            nearest_protected_distance=distance,
            prospective_radius=radius,
            # Tiny distances are ambiguous, never inflated to allow an update.
            separable=distance > math.sqrt(torch.finfo(torch.float64).eps),
        )
