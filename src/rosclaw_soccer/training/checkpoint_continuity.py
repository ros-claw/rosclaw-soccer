"""Validate adjacent learner snapshots, independently of optimizer replay.

Reproducing one update from a supplied optimizer state does not establish that
the supplied state actually came from the preceding saved update. This check
closes that separate link. It performs no loading, mutation or policy approval.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any


def verify_checkpoint_continuity(
    preceding_resume: dict[str, Any],
    following_before_update: dict[str, Any],
    *,
    expected_preceding_generation: int,
) -> None:
    """Require exact model, optimizer and Torch-RNG continuity across updates.

    Inputs are already safely decoded, caller-owned snapshots. Callers retain
    hashes, source/data commitments and initialization checks separately. A
    declared optimizer/RNG reset starts a NEW protocol; it cannot silently pass
    as an adjacent update. Torch is optional until this function is invoked.
    Only bounded plain containers (including Torch's OrderedDict state and its
    metadata), scalar metadata and finite strided
    tensors are accepted; tensors must agree in shape, dtype and device.
    """
    if (
        type(expected_preceding_generation) is not int
        or not 0 <= expected_preceding_generation <= 1_000_000
        or type(preceding_resume) is not dict
        or type(following_before_update) is not dict
        or set(preceding_resume) != {"model", "optimizer", "rng_snapshot", "generation"}
        or set(following_before_update) != {"model", "optimizer", "rng_snapshot"}
        or type(preceding_resume["generation"]) is not int
        or preceding_resume["generation"] != expected_preceding_generation
    ):
        raise ValueError("adjacent checkpoint envelope or generation mismatch")
    import math

    import torch

    visited = 0

    def equal(left: Any, right: Any, depth: int) -> None:
        nonlocal visited
        visited += 1
        if depth > 24 or visited > 100_000:
            raise ValueError("checkpoint continuity structure exceeds bounds")
        if isinstance(left, torch.Tensor):
            if (
                not isinstance(right, torch.Tensor)
                or left.layout != torch.strided
                or right.layout != torch.strided
                or left.dtype != right.dtype
                or left.device != right.device
                or left.shape != right.shape
                or left.requires_grad
                or right.requires_grad
                or not bool(torch.isfinite(left).all())
                or not bool(torch.isfinite(right).all())
                or not torch.equal(left, right)
            ):
                raise ValueError("checkpoint continuity tensor mismatch or invalid value")
        elif type(left) in (dict, OrderedDict):
            if (
                type(right) is not type(left)
                or left.keys() != right.keys()
                or any(type(key) not in (str, int) for key in left)
                or any(type(key) not in (str, int) for key in right)
            ):
                raise ValueError("checkpoint continuity mapping mismatch")
            for key in left:
                equal(left[key], right[key], depth + 1)
            if type(left) is OrderedDict:
                equal(left.__dict__, right.__dict__, depth + 1)
        elif type(left) in (list, tuple):
            if type(right) is not type(left) or len(left) != len(right):
                raise ValueError("checkpoint continuity sequence mismatch")
            for a, b in zip(left, right, strict=True):
                equal(a, b, depth + 1)
        elif type(left) in (str, int, bool, float, type(None)):
            if (
                type(left) is not type(right)
                or left != right
                or (type(left) is float and not math.isfinite(left))
            ):
                raise ValueError("checkpoint continuity scalar mismatch or invalid value")
        else:
            raise ValueError("unsupported checkpoint continuity value")

    for key in ("model", "optimizer", "rng_snapshot"):
        if type(preceding_resume[key]) not in (dict, OrderedDict) or type(
            following_before_update[key]
        ) not in (dict, OrderedDict):
            raise ValueError("checkpoint components must be plain mappings")
        equal(preceding_resume[key], following_before_update[key], 0)
