"""Value-only snapshots of the qualified G1 locomotion LSTM, for private replay.

This is not a whole-controller or world checkpoint: previous actions, reflection
frame, command construction and physical integration state remain caller-owned.
No pickle, device handles, optimizer state or writable tensor aliases are exported.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any, cast

import numpy as np


@dataclass(frozen=True)
class LocomotionMemory:
    policy_hash: str
    hidden: bytes
    cell: bytes

    def __post_init__(self) -> None:
        if not isinstance(self.policy_hash, str) or not re.fullmatch(
            r"sha256:[0-9a-f]{64}", self.policy_hash
        ):
            raise ValueError("qualified policy SHA256 required")
        for value in (self.hidden, self.cell):
            if type(value) is not bytes or len(value) != 256 * 4:
                raise ValueError("immutable float32 LSTM memory required")
            if not np.isfinite(np.frombuffer(value, dtype="<f4")).all():
                raise ValueError("finite LSTM memory required")

    @property
    def state_hash(self) -> str:
        return (
            "sha256:"
            + hashlib.sha256(
                b"g1-locomotion-memory.v1\0"
                + bytes.fromhex(self.policy_hash.removeprefix("sha256:"))
                + self.hidden
                + self.cell
            ).hexdigest()
        )


def _states(model: Any) -> tuple[Any, Any]:
    import torch

    states = tuple(getattr(model, name, None) for name in ("hidden_state", "cell_state"))
    for value in states:
        if (
            not isinstance(value, torch.Tensor)
            or tuple(value.shape) != (1, 1, 256)
            or value.dtype != torch.float32
            or value.device.type != "cpu"
            or not bool(torch.isfinite(value).all())
        ):
            raise ValueError("qualified finite CPU float32 (1,1,256) LSTM state required")
    hidden, cell = cast(Any, states[0]), cast(Any, states[1])
    if hidden.untyped_storage().data_ptr() == cell.untyped_storage().data_ptr():
        raise ValueError("hidden and cell must have independent storage")
    return hidden, cell


def capture_locomotion_memory(model: Any, *, policy_hash: str) -> LocomotionMemory:
    """Copy post-inference memory without advancing or detaching the live model."""
    hidden, cell = _states(model)
    return LocomotionMemory(
        policy_hash,
        hidden.detach().numpy().astype("<f4", copy=True).tobytes(),
        cell.detach().numpy().astype("<f4", copy=True).tobytes(),
    )


def restore_private_locomotion_memory(
    model: Any,
    snapshot: LocomotionMemory,
    *,
    policy_hash: str,
    private_replay: bool,
) -> None:
    """Restore only an independently loaded research model; never the live actor.

    Explicit ownership is a trusted caller contract, not a security sandbox.
    The caller must hash the loaded artifact, not supply an inferred model ID.
    """
    import torch

    if private_replay is not True:
        raise ValueError("explicit private replay ownership required")
    if not isinstance(snapshot, LocomotionMemory) or snapshot.policy_hash != policy_hash:
        raise ValueError("locomotion policy binding differs")
    snapshot.__post_init__()
    hidden, cell = _states(model)
    replacements = [
        torch.from_numpy(np.frombuffer(value, dtype="<f4").copy().reshape(1, 1, 256))
        for value in (snapshot.hidden, snapshot.cell)
    ]
    originals = [hidden.detach().clone(), cell.detach().clone()]
    with torch.no_grad():
        try:
            hidden.copy_(replacements[0])
            cell.copy_(replacements[1])
        except Exception:
            hidden.copy_(originals[0])
            cell.copy_(originals[1])
            raise
