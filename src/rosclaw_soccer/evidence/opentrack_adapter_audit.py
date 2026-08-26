"""Parameter-isolation audit for external OpenTrack adapter checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.sim.contracts import hash_json


def compare_opentrack_policy_parameters(
    *,
    parent_policy: Mapping[str, Mapping[str, Any]],
    candidate_policy: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Prove the inherited MLP is immutable and enumerate residual weights."""

    parent_hidden = {
        key: value for key, value in parent_policy.items() if key.startswith("hidden_")
    }
    candidate_hidden = {
        key: value for key, value in candidate_policy.items() if key.startswith("hidden_")
    }
    candidate_adapter = {
        key: value for key, value in candidate_policy.items() if key.startswith("adapter_")
    }
    if not parent_hidden or not candidate_adapter:
        raise ValueError("OpenTrack audit requires inherited hidden and residual adapter layers")
    if set(parent_hidden) != set(candidate_hidden):
        raise ValueError("OpenTrack candidate changed the inherited hidden-layer topology")

    maximum_drift = 0.0
    frozen_scalars = 0
    for layer_name in sorted(parent_hidden):
        parent_layer = parent_hidden[layer_name]
        candidate_layer = candidate_hidden[layer_name]
        if set(parent_layer) != set(candidate_layer):
            raise ValueError("OpenTrack candidate changed an inherited layer parameter set")
        for parameter_name in sorted(parent_layer):
            before = np.asarray(parent_layer[parameter_name])
            after = np.asarray(candidate_layer[parameter_name])
            if before.shape != after.shape:
                raise ValueError("OpenTrack candidate changed an inherited parameter shape")
            frozen_scalars += int(before.size)
            maximum_drift = max(
                maximum_drift,
                float(np.max(np.abs(before.astype(np.float64) - after.astype(np.float64)))),
            )
    trainable_scalars = sum(
        int(np.asarray(value).size)
        for layer in candidate_adapter.values()
        for value in layer.values()
    )
    if min(frozen_scalars, trainable_scalars) <= 0 or not math.isfinite(maximum_drift):
        raise ValueError("OpenTrack parameter audit produced an empty or non-finite scope")
    return {
        "frozen_base_hash_before": _parameter_hash(parent_hidden),
        "frozen_base_hash_after": _parameter_hash(candidate_hidden),
        "maximum_frozen_parameter_drift": maximum_drift,
        "examined_frozen_parameter_count": frozen_scalars,
        "examined_trainable_parameter_count": trainable_scalars,
        "frozen_layers": sorted(parent_hidden),
        "trainable_layers": sorted(candidate_adapter),
    }


def audit_opentrack_adapter_checkpoint(
    *, parent_checkpoint: Path, candidate_checkpoint: Path, output_path: Path
) -> dict[str, Any]:
    """Load two Orbax checkpoints and persist a content-addressed audit."""

    parent = parent_checkpoint.expanduser().resolve()
    candidate = candidate_checkpoint.expanduser().resolve()
    output = output_path.expanduser().resolve()
    if not parent.is_dir() or not candidate.is_dir():
        raise FileNotFoundError("OpenTrack parent and candidate checkpoints must exist")
    if parent == candidate:
        raise ValueError("OpenTrack parameter audit requires distinct checkpoints")
    if output.exists():
        raise ValueError("OpenTrack parameter audit refuses to overwrite evidence")
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    checkpoint_module = importlib.import_module("brax.training.agents.ppo.checkpoint")
    parent_params = checkpoint_module.load(str(parent))
    candidate_params = checkpoint_module.load(str(candidate))
    if len(parent_params) < 2 or len(candidate_params) < 5:
        raise ValueError("OpenTrack checkpoint parameter trees are incomplete")
    result = compare_opentrack_policy_parameters(
        parent_policy=parent_params[1]["params"],
        candidate_policy=candidate_params[1]["params"],
    )
    adapter_scalars = int(result["examined_trainable_parameter_count"])
    trainable_scope_counts = {
        "policy_adapter": adapter_scalars,
        "value_network": _tree_scalar_count(candidate_params[2]),
        "world_model": _tree_scalar_count(candidate_params[3]),
        "history_encoder": _tree_scalar_count(candidate_params[4]),
    }
    if any(count <= 0 for count in trainable_scope_counts.values()):
        raise ValueError("OpenTrack audit found an empty declared trainable scope")
    result["trainable_scope_counts"] = trainable_scope_counts
    result["examined_trainable_parameter_count"] = sum(trainable_scope_counts.values())
    if not candidate.name.isdigit() or int(candidate.name) <= 0:
        raise ValueError("OpenTrack candidate checkpoint directory must encode world steps")
    result["candidate_world_steps"] = int(candidate.name)
    report: dict[str, Any] = {
        "schema_version": "rosclaw_soccer.opentrack_parameter_isolation.v1",
        "parent_checkpoint_hash": _tree_hash(parent),
        "candidate_checkpoint_hash": _tree_hash(candidate),
        **result,
        "frozen_base_unchanged": bool(
            result["frozen_base_hash_before"] == result["frozen_base_hash_after"]
            and result["maximum_frozen_parameter_drift"] == 0.0
        ),
        "activation_ceiling": "SIM_ONLY",
        "hardware_command_sent": False,
    }
    report["report_hash"] = hash_json(report)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, output)
    return report


def _parameter_hash(layers: Mapping[str, Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for layer_name in sorted(layers):
        for parameter_name in sorted(layers[layer_name]):
            value = np.asarray(layers[layer_name][parameter_name])
            digest.update(layer_name.encode())
            digest.update(parameter_name.encode())
            digest.update(value.dtype.str.encode())
            digest.update(json.dumps(value.shape).encode())
            digest.update(np.ascontiguousarray(value).tobytes())
    return "sha256:" + digest.hexdigest()


def _tree_scalar_count(tree: Any) -> int:
    if isinstance(tree, Mapping):
        return sum(_tree_scalar_count(value) for value in tree.values())
    if isinstance(tree, (tuple, list)):
        return sum(_tree_scalar_count(value) for value in tree)
    value = np.asarray(tree)
    return int(value.size)


def _tree_hash(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(candidate for candidate in root.rglob("*") if candidate.is_file()):
        digest.update(path.relative_to(root).as_posix().encode())
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit OpenTrack adapter parameter isolation")
    parser.add_argument("--parent-checkpoint", required=True, type=Path)
    parser.add_argument("--candidate-checkpoint", required=True, type=Path)
    parser.add_argument("--output-path", required=True, type=Path)
    args = parser.parse_args()
    report = audit_opentrack_adapter_checkpoint(
        parent_checkpoint=args.parent_checkpoint,
        candidate_checkpoint=args.candidate_checkpoint,
        output_path=args.output_path,
    )
    print(
        json.dumps(
            {"report_hash": report["report_hash"], "frozen": report["frozen_base_unchanged"]}
        )
    )


if __name__ == "__main__":
    main()


__all__ = ["audit_opentrack_adapter_checkpoint", "compare_opentrack_policy_parameters"]
