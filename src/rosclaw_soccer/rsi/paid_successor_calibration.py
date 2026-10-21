"""Training-only conservative threshold calibration, not physical qualification."""

import copy
import math
from typing import Any

import torch
from torch import nn


def calibrate_successor_gate(
    gate: nn.Sequential, negative_features: torch.Tensor, *, margin: float = 0.1
) -> dict[str, Any]:
    if (
        not math.isfinite(margin)
        or not 0 < margin <= 1
        or negative_features.ndim != 2
        or negative_features.shape[1] != 288
        or not len(negative_features)
        or negative_features.dtype != torch.float32
        or not torch.isfinite(negative_features).all()
        or len(gate) != 3
        or not isinstance(gate[2], nn.Linear)
        or gate[2].bias is None
        or gate[2].bias.shape != (1,)
    ):
        raise ValueError("finite complete negative288 support and scalar output gate required")
    with torch.no_grad():
        x = negative_features.detach().cpu()
        cpu = copy.deepcopy(gate).cpu().eval()
        before = cpu(x)
        if before.shape != (len(x), 1) or not torch.isfinite(before).all():
            raise ValueError("finite scalar score for every retained state required")
        maximum = float(before.max())
        shift = max(0.0, maximum + margin + 0.0001)
        bias_before = float(gate[2].bias[0])
        gate[2].bias.sub_(shift)
        cpu = copy.deepcopy(gate).cpu().eval()
        after = cpu(x)
        final = float(after.max())
        if not torch.isfinite(after).all() or final > -margin:
            raise ValueError("CPU complete-support conservative gate calibration failed")
    return dict(
        maximum_negative_before_CPU=maximum,
        maximum_negative_after_CPU=final,
        output_bias_shift=shift,
        bias_before=bias_before,
        bias_after=float(gate[2].bias[0]),
        negative_support_rows=len(x),
        CPU_gate_forward_rows=2 * len(x),
        margin=margin,
        only_output_bias_shifted=True,
        all_negative_gate_rejected=True,
        not_physical_qualification=True,
    )
