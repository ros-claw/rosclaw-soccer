"""Experimental functional retention projection for the PAiD contact gate.

Training-only, not a runtime case router or physical safety certification.
Only the four gate tensors can be interpolated; frozen motor weights and the
plastic joint-delta head are outside this operation. Adam moments are NOT
rewound: a caller must record projected/partial/rejected parameter updates.
"""

import math
from typing import Any

import torch
from torch import nn


def project_contact_gate(
    gate: nn.Module,
    previous: dict[str, torch.Tensor],
    retention_features: torch.Tensor,
    *,
    margin: float = 0.1,
    max_halvings: int = 8,
) -> dict[str, Any]:
    """Keep every finite retained-state logit <= -margin, not just its mean.

    Old contexts must be represented by causally recomputed frozen-base288
    features. At inference this negative margin implies an exact-zero motor
    correction. Feasibility is empirical on these states, NOT new-world skill
    proof. A full old-policy physical exam remains mandatory.
    """
    if (
        not math.isfinite(margin)
        or not 0 < margin <= 1
        or type(max_halvings) is not int
        or not 0 <= max_halvings <= 8
    ):
        raise ValueError("finite positive margin and bounded backtracking required")
    params = dict(gate.named_parameters())
    shapes = {"0.weight": (64, 288), "0.bias": (64,), "2.weight": (1, 64), "2.bias": (1,)}
    if set(params) != set(shapes) or set(previous) != set(shapes):
        raise ValueError("only the declared four contact-gate tensors may be projected")
    if (
        retention_features.ndim != 2
        or retention_features.shape[1] != 288
        or not len(retention_features)
        or retention_features.dtype != torch.float32
        or not torch.isfinite(retention_features).all()
    ):
        raise ValueError("all finite original retained-state288 features required")
    for k, p in params.items():
        old = previous[k]
        if (
            p.shape != shapes[k]
            or p.dtype != torch.float32
            or p.device != retention_features.device
            or not torch.isfinite(p).all()
            or not isinstance(old, torch.Tensor)
            or old.shape != p.shape
            or old.dtype != p.dtype
            or old.device != p.device
            or not torch.isfinite(old).all()
        ):
            raise ValueError("finite same-device exact old/proposed gate tensors required")
    proposal = {k: p.detach().clone() for k, p in params.items()}
    evaluations = 0
    with torch.no_grad():

        def maximum() -> float:
            nonlocal evaluations
            result = gate(retention_features)
            evaluations += 1
            if result.shape != (len(retention_features), 1) or not torch.isfinite(result).all():
                raise ValueError("finite scalar gate score per retained state required")
            return float(result.max())

        def restore(fraction: float) -> None:
            for k, p in params.items():
                p.copy_(
                    previous[k]
                    if fraction == 0
                    else proposal[k]
                    if fraction == 1
                    else previous[k] + fraction * (proposal[k] - previous[k])
                )

        value = maximum()
        fraction = 1.0
        if value > -margin:
            restore(0)
            old_max = maximum()
            if old_max > -margin:
                raise ValueError(
                    "previous gate was not feasible; preserve failure, no blind repair"
                )
            fraction = 0.0
            value = old_max
            for power in range(1, max_halvings + 1):
                trial = 2.0 ** (-power)
                restore(trial)
                score = maximum()
                if score <= -margin:
                    fraction, value = trial, score
                    break
            if fraction == 0:
                restore(0)
        return dict(
            accepted_fraction=fraction,
            maximum_retained_logit=value,
            gate_evaluations=evaluations,
            gate_forward_rows=evaluations * len(retention_features),
            gate_update_rejected=fraction == 0,
            only_gate_parameters_projected=True,
            optimizer_moments_rewound=False,
            not_physical_qualification=True,
        )
