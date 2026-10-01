"""Bound resident sampling-view memory without changing serialized policies.

The original sampling constructors remain authoritative and unchanged. This
collector-only helper releases each identical copied mean after construction;
serialization still embeds the COMPLETE mean in EVERY sampling artifact.
"""

from collections.abc import Callable, Iterable
from typing import Any

from rosclaw_soccer.sim.contracts import hash_json


def bounded_views(
    factory: Callable[..., dict[str, Any]],
    base: dict[str, Any],
    seeds: Iterable[int],
) -> list[dict[str, Any]]:
    original = hash_json(base)
    views = []
    for seed in seeds:
        view = factory(base, seed=seed, std=0.1)
        if "mean_model" in view:
            if (
                view["schema"]
                not in (
                    "soccer.rsi.output_memory_step_sampling.v1",
                    "soccer.rsi.smooth_memory_sampling.v1",
                )
                or view["mean_model"] != base
            ):
                raise ValueError("sampling constructor changed the complete declared mean")
            expected = view["model_hash"]
            view["mean_model"] = base
            if hash_json({k: v for k, v in view.items() if k != "model_hash"}) != expected:
                raise ValueError("sharing changed serialized sampling policy identity")
        views.append(view)
    if hash_json(base) != original:
        raise ValueError("sampling construction mutated its immutable mean")
    return views
