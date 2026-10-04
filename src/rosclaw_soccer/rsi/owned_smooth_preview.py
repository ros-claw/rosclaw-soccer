"""Owned, source-pinned SIM-only mean validation and byte-identical previews.

The ORIGINAL mean preview validates the entire model once. Each subsequent
sampling view must contain the exact canonical mean bytes, valid bounded
sampling scalars, unchanged source, and its complete original model hash.
This does not cache an untrusted object by id or skip integrity checks.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from rosclaw.growth import correlated_exploration

from rosclaw_soccer.rsi import smooth_memory_motor
from rosclaw_soccer.rsi.contact_motor_phase import make_policy
from rosclaw_soccer.rsi.kernel_guarded_step_network import FLAGS
from rosclaw_soccer.sim.contracts import hash_bytes


def _json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode()


def _with_mean(value: Any, path: tuple[str, ...], mean_path: tuple[str, ...], mean: bytes) -> bytes:
    if path == mean_path:
        return mean
    if type(value) is dict:
        return (
            b"{"
            + b",".join(
                _json(key) + b":" + _with_mean(value[key], (*path, key), mean_path, mean)
                for key in sorted(value)
            )
            + b"}"
        )
    return _json(value)


class OwnedSmoothPreview:
    """One fully validated mean; accepts only ordinary JSON sampling documents.

    Mean bytes are compared again on EVERY call, including numeric JSON types.
    Cached bytes are injected only at the one declared nested mean path. No
    mutable caller object, source drift or authority flag is grandfathered.
    """

    def __init__(self, mean_model: dict[str, Any]) -> None:
        if type(mean_model) is not dict or mean_model.get("schema") != smooth_memory_motor.SCHEMA:
            raise ValueError("owned preview requires an ordinary smooth mean")
        mean = copy.deepcopy(mean_model)
        original = smooth_memory_motor.make_preview(mean)
        self._mean = _json(mean)
        self._proof = {k: v for k, v in original["step_motor_proof"].items() if k != "model"}
        self._memory_proof = copy.deepcopy(original["smooth_memory_motor_proof"])
        self._static = dict(
            schema=smooth_memory_motor.SAMPLING_SCHEMA,
            activation_ceiling="SIM_ONLY",
            training_only=True,
            source_hash=hash_bytes(Path(smooth_memory_motor.__file__).read_bytes()),
            **dict.fromkeys(FLAGS, False),
        )
        files = list(Path(smooth_memory_motor.__file__).parent.glob("*.py"))
        files += list(Path(correlated_exploration.__file__).parent.glob("*.py"))
        files += [Path(__file__), Path(__file__).parents[1] / "sim/contracts.py"]
        self._pins = {str(p): hash_bytes(p.read_bytes()) for p in files}

    def _stable(self) -> None:
        if any(hash_bytes(Path(p).read_bytes()) != h for p, h in self._pins.items()):
            raise ValueError("owned preview dependency source changed")

    def preview(self, wrapped: dict[str, Any]) -> dict[str, Any]:
        self._stable()
        keys = set(self._static) | {"mean_model", "seed", "std_raw", "rho", "model_hash"}
        if type(wrapped) is not dict or set(wrapped) != keys:
            raise ValueError("complete exact owned sampling document required")
        if any(
            (
                wrapped[k] is not v
                if type(v) is bool
                else type(wrapped[k]) is not type(v) or wrapped[k] != v
            )
            for k, v in self._static.items()
        ):
            raise ValueError("owned sampling source/authority differs")
        if type(wrapped["mean_model"]) is not dict or _json(wrapped["mean_model"]) != self._mean:
            raise ValueError("owned sampling canonical mean differs")
        seed, std, rho = (wrapped[k] for k in ("seed", "std_raw", "rho"))
        if (
            type(seed) is not int
            or not 0 <= seed < 2**32
            or type(std) not in (int, float)
            or not np.isfinite(std)
            or not 0.01 <= std <= 0.15
            or type(rho) not in (int, float)
            or not np.isfinite(rho)
            or not 0 <= rho <= 0.95
            or correlated_exploration.conditional_scale(std, rho, first=False) < 0.01
        ):
            raise ValueError("bounded stationary owned sampling required")
        content = {k: v for k, v in wrapped.items() if k != "model_hash"}
        digest = (
            "sha256:"
            + hashlib.sha256(_with_mean(content, (), ("mean_model",), self._mean)).hexdigest()
        )
        if wrapped["model_hash"] != digest:
            raise ValueError("owned sampling original model hash differs")
        policy = make_policy(np.zeros((3, 12)), 0.25, digest)
        policy["execution_profile"] = "causal_per_frame_neural_residual"
        policy["step_motor_proof"] = dict(
            self._proof, model=wrapped, qualification="UNQUALIFIED_SIM_TRAINING"
        )
        policy["smooth_memory_motor_proof"] = copy.deepcopy(self._memory_proof)
        policy.pop("policy_hash")
        encoded = _with_mean(policy, (), ("step_motor_proof", "model", "mean_model"), self._mean)
        policy["policy_hash"] = "sha256:" + hashlib.sha256(encoded).hexdigest()
        return policy

    def validate_preview(self, policy: dict[str, Any]) -> dict[str, Any]:
        if type(policy) is not dict or type(policy.get("step_motor_proof")) is not dict:
            raise ValueError("complete owned sampling preview required")
        expected = self.preview(policy["step_motor_proof"].get("model"))
        path = ("step_motor_proof", "model", "mean_model")
        if set(policy) != set(expected) or _with_mean(policy, (), path, self._mean) != _with_mean(
            expected, (), path, self._mean
        ):
            raise ValueError("owned sampling original preview integrity differs")
        # Separate scalar container before returning to the numeric decoder.
        return dict(expected["step_motor_proof"]["model"])
