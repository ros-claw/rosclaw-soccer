"""SIM research fork of one SONIC tracker, sharing weights but no mutable history.

This is not model qualification, a complete controller fork, a simulator, or
an action entry. In particular, the caller must separately fork task-space
and residual policy state before claiming a full closed-loop counterfactual.
"""

from typing import Any

from rosclaw_soccer.providers.g1.sonic_torch import FrozenSonicG1Torch
from rosclaw_soccer.providers.g1.sonic_vector import BatchedSonicTracker

FIELDS = frozenset(
    {
        "low_latency_legacy_encoder_layout",
        "_capture_neural_observation",
        "_neural_observation",
        "_torch",
        "model",
        "reference",
        "count",
        "stride",
        "native_velocity",
        "default",
        "kp",
        "kd",
        "scale",
        "_to_isaac",
        "_to_mujoco",
        "_history",
        "_next_frame",
        "_pending_observation",
        "action",
    }
)
MAX_OWNED_TENSOR_BYTES = 8 * 1024 * 1024


def fork_sonic_tracker(tracker: BatchedSonicTracker) -> BatchedSonicTracker:
    """Fork an updated single-lane tracker at its pending-observation boundary.

    Only the exact frozen-model implementation and torch module are shared.
    Explicit fields fail closed if the tracker later acquires unreviewed state.
    No neural call, reference refresh, clock advance, reset, or physics occurs.
    """
    import torch

    if (
        type(tracker) is not BatchedSonicTracker
        or set(vars(tracker)) != FIELDS
        or type(tracker.model) is not FrozenSonicG1Torch
        or tracker._torch is not torch
        or type(tracker.count) is not int
        or tracker.count != 1
        or type(tracker._next_frame) is not int
        or not 1 <= tracker._next_frame < 300
        or tracker._pending_observation is not True
        or len(tracker._history) != 10
    ):
        raise ValueError(
            "exact updated single-lane frozen SONIC boundary and reviewed fields required"
        )
    if (
        not isinstance(tracker.action, torch.Tensor)
        or tracker.action.shape != (1, 29)
        or tracker.action.dtype != torch.float32
        or any(
            type(entry) is not tuple
            or len(entry) != 5
            or any(
                not isinstance(value, torch.Tensor)
                or value.shape != (1, width)
                or value.dtype != torch.float32
                for value, width in zip(entry, (3, 29, 29, 29, 3), strict=True)
            )
            for entry in tracker._history
        )
    ):
        raise ValueError("complete canonical measured history and current action required")
    size = 0

    def clone(value: Any) -> Any:
        nonlocal size
        if isinstance(value, torch.Tensor):
            size += value.numel() * value.element_size()
            if (
                size > MAX_OWNED_TENSOR_BYTES
                or value.requires_grad
                or not bool(torch.isfinite(value).all())
            ):
                raise ValueError("bounded finite frozen-inference tracker tensors required")
            return value.detach().clone()
        if type(value) in (bool, int, float) or value is None:
            return value
        if type(value) is list:
            return [clone(v) for v in value]
        if type(value) is tuple:
            return tuple(clone(v) for v in value)
        if type(value) is dict and all(type(k) is str for k in value):
            return {k: clone(v) for k, v in value.items()}
        raise ValueError("unreviewed mutable tracker state cannot be shared")

    fork = object.__new__(BatchedSonicTracker)
    for key, value in vars(tracker).items():
        setattr(fork, key, value if key in ("model", "_torch") else clone(value))
    return fork
