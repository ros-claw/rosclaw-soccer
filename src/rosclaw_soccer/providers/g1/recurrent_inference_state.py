"""Bound graph lifetime without resetting a frozen recurrent policy's memory.

This helper is for a simulation-owned, frozen inference model only. It does
not select an action, change tensor values, or grant hardware/training access.
The caller must exclude this model from every optimizer and must not attempt
backpropagation through its persistent inference state.
"""

from __future__ import annotations

from typing import Any, cast


def detach_frozen_recurrent_state(policy: Any, *, frozen_inference: bool) -> None:
    """Detach the qualified LSTM state pair; preserve the forward numeric path.

    Unlike switching eval/no-grad modes, detachment does not change the
    export's forward kernels. Values, device, dtype and storage are preserved;
    only the persistent autograd graph is discarded. Failed input validation
    does not partially replace the state pair.
    """
    import torch

    if frozen_inference is not True:
        raise ValueError("explicit frozen recurrent inference ownership required")
    states = tuple(getattr(policy, name, None) for name in ("hidden_state", "cell_state"))
    if any(
        not isinstance(value, torch.Tensor)
        or value.ndim != 3
        or not 1 <= value.shape[0] <= 4
        or not 1 <= value.shape[1] <= 4096
        or not 1 <= value.shape[2] <= 4096
        or value.numel() > 4_194_304
        or value.dtype not in {torch.float32, torch.float64}
        or not bool(torch.isfinite(value).all())
        for value in states
    ):
        raise ValueError("bounded finite LSTM hidden/cell state pair required")
    hidden, cell = cast(Any, states[0]), cast(Any, states[1])
    if hidden.shape != cell.shape or hidden.device != cell.device or hidden.dtype != cell.dtype:
        raise ValueError("LSTM state pair layout differs")
    detached_hidden, detached_cell = hidden.detach(), cell.detach()
    try:
        policy.hidden_state = detached_hidden
        policy.cell_state = detached_cell
    except Exception:
        policy.hidden_state = hidden
        policy.cell_state = cell
        raise
