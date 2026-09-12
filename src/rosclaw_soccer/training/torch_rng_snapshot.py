"""Explicit, CPU-stored RNG snapshots for an exclusively owned learner process."""

import os
import re
from typing import Any


def _devices(devices: tuple[str, ...]) -> None:
    if (
        type(devices) is not tuple
        or len(devices) > 16
        or any(
            type(device) is not str or re.fullmatch(r"cuda:(?:0|[1-9][0-9]{0,2})", device) is None
            for device in devices
        )
        or len(set(devices)) != len(devices)
    ):
        raise ValueError("explicit distinct CUDA device strings required")


def capture_torch_rng(*, cuda_devices: tuple[str, ...] = ()) -> dict[str, Any]:
    """Capture CPU and only the requested CUDA generators, never all devices.

    All tensors remain on CPU for weights-only checkpoint loading. Logical
    device mapping is bound to CUDA_VISIBLE_DEVICES when CUDA is requested.
    This records randomness, not optimizer ownership or simulation authority.
    """
    import torch

    _devices(cuda_devices)
    return {
        "schema": "rosclaw_soccer.torch_rng.v1",
        "cpu": torch.get_rng_state().clone(),
        "cuda": {device: torch.cuda.get_rng_state(device).cpu().clone() for device in cuda_devices},
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES") if cuda_devices else None,
    }


def restore_torch_rng(snapshot: Any, *, cuda_devices: tuple[str, ...] = ()) -> None:
    """Validate every generator in isolation before changing default generators.

    Caller must exclude concurrent RNG users. Device/runtime failures are not
    a transactional checkpoint guarantee; retain the complete prior checkpoint.
    No device remapping, dtype conversion or missing-state fallback is implicit.
    """
    import torch

    _devices(cuda_devices)
    if (
        type(snapshot) is not dict
        or set(snapshot) != {"schema", "cpu", "cuda", "cuda_visible_devices"}
        or snapshot["schema"] != "rosclaw_soccer.torch_rng.v1"
        or type(snapshot["cuda"]) is not dict
        or set(snapshot["cuda"]) != set(cuda_devices)
        or snapshot["cuda_visible_devices"]
        != (os.environ.get("CUDA_VISIBLE_DEVICES") if cuda_devices else None)
    ):
        raise ValueError("complete RNG snapshot with the same explicit device mapping required")
    generators = {"cpu": snapshot["cpu"], **snapshot["cuda"]}
    for device, state in generators.items():
        if (
            not isinstance(state, torch.Tensor)
            or state.device.type != "cpu"
            or state.dtype != torch.uint8
            or state.ndim != 1
            or not 1 <= state.numel() <= 1048576
            or not state.is_contiguous()
        ):
            raise ValueError("bounded contiguous CPU uint8 RNG states required")
        # A private generator checks backend-specific state contents without
        # partially restoring any default generator on malformed input.
        torch.Generator(device=device).set_state(state)
    torch.set_rng_state(snapshot["cpu"])
    for device in cuda_devices:
        torch.cuda.set_rng_state(snapshot["cuda"][device], device=device)
