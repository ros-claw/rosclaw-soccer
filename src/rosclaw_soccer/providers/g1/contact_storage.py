"""Validate padded simulator contact storage without dynamic-size GPU indexing."""

from typing import Any


def validate_active_contact_storage(
    normal_force: Any, world_id: Any, active_slot: Any, *, environment_count: int
) -> None:
    """Check every occupied slot; ignore uninitialized padding, never sensor faults.

    Equivalent value acceptance to checking ``force[active_slot]`` and
    ``world[active_slot]``, but uses fixed-shape reductions and one host decision.
    No simulator state, inputs, step counters or contact labels are modified.
    This is a storage check, not a proof of contact causality or robot safety.
    """
    import torch

    if (
        type(environment_count) is not int
        or not 1 <= environment_count <= 4096
        or any(
            not isinstance(value, torch.Tensor) or value.ndim != 1 or value.layout != torch.strided
            for value in (normal_force, world_id, active_slot)
        )
        or normal_force.dtype not in (torch.float32, torch.float64)
        or world_id.dtype not in (torch.int32, torch.int64)
        or active_slot.dtype != torch.bool
        or any(
            value.shape != normal_force.shape or value.device != normal_force.device
            for value in (world_id, active_slot)
        )
    ):
        raise ValueError("aligned finite-force storage, integer worlds and boolean slots required")
    occupied_valid = torch.isfinite(normal_force) & (world_id >= 0) & (world_id < environment_count)
    if not bool((occupied_valid | ~active_slot).all()):
        raise FloatingPointError("invalid physical contact force or world identity")
