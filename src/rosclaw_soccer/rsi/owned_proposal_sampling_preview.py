"""Source-pinned original proposal previews from one fully validated mean.

This is explicit offline compilation, not a validation bypass by object id.
Every call checks complete canonical mean bytes, sampling fields and seals.
The unmodified reference remains the independent qualification path.
"""

import copy
import hashlib
from pathlib import Path
from typing import Any

from rosclaw.growth import correlated_exploration

from rosclaw_soccer.rsi import proposal_sampling_motor
from rosclaw_soccer.rsi.owned_smooth_preview import _json, _with_mean
from rosclaw_soccer.sim.contracts import hash_bytes


class OwnedProposalSamplingPreview:
    """Private immutable mean bytes and an original, source-bound template."""

    def __init__(self, mean_model: dict[str, Any]) -> None:
        original = proposal_sampling_motor.make_preview(
            proposal_sampling_motor.make_sampling_view(copy.deepcopy(mean_model), seed=0)
        )
        view = original["step_motor_proof"]["model"]
        self._mean = _json(view["mean_model"])
        self._static = {
            k: v for k, v in view.items() if k not in ("mean_model", "seed", "model_hash")
        }
        self._template = copy.deepcopy({k: v for k, v in original.items() if k != "policy_hash"})
        self._template["step_motor_proof"].pop("model")
        self._template["proposal_sampling_motor_proof"].pop("sampling_model_hash")
        files = list(Path(proposal_sampling_motor.__file__).parent.glob("*.py"))
        files += list(Path(correlated_exploration.__file__).parent.glob("*.py"))
        files += [Path(__file__).parents[1] / "sim/contracts.py"]
        self._pins = {str(p): hash_bytes(p.read_bytes()) for p in files}

    def _stable(self) -> None:
        if any(hash_bytes(Path(p).read_bytes()) != h for p, h in self._pins.items()):
            raise ValueError("owned proposal preview dependency source changed")

    def preview(self, wrapped: dict[str, Any]) -> dict[str, Any]:
        self._stable()
        if type(wrapped) is not dict or set(wrapped) != set(self._static) | {
            "mean_model",
            "seed",
            "model_hash",
        }:
            raise ValueError("complete exact owned proposal sampling document required")
        # Canonical equality distinguishes false/0, true/1, signed zero, and
        # float/int. An unchanged caller-supplied hash grants no trust.
        if _json({k: wrapped[k] for k in self._static}) != _json(self._static):
            raise ValueError("owned proposal sampling fields/source/authority differ")
        seed = wrapped["seed"]
        if type(seed) is not int or not 0 <= seed < 2**32:
            raise ValueError("bounded integer sampling seed required")
        owned = copy.deepcopy(wrapped)
        if type(owned["mean_model"]) is not dict or _json(owned["mean_model"]) != self._mean:
            raise ValueError("owned proposal sampling canonical mean differs")
        content = {k: v for k, v in owned.items() if k != "model_hash"}
        digest = (
            "sha256:"
            + hashlib.sha256(_with_mean(content, (), ("mean_model",), self._mean)).hexdigest()
        )
        if type(owned["model_hash"]) is not str or owned["model_hash"] != digest:
            raise ValueError("owned proposal sampling original model hash differs")
        policy = copy.deepcopy(self._template)
        policy["step_motor_proof"]["model"] = owned
        policy["proposal_sampling_motor_proof"]["sampling_model_hash"] = digest
        encoded = _with_mean(policy, (), ("step_motor_proof", "model", "mean_model"), self._mean)
        policy["policy_hash"] = "sha256:" + hashlib.sha256(encoded).hexdigest()
        return policy

    def validate_preview(self, policy: dict[str, Any]) -> dict[str, Any]:
        if type(policy) is not dict or type(policy.get("step_motor_proof")) is not dict:
            raise ValueError("complete owned proposal sampling preview required")
        expected = self.preview(policy["step_motor_proof"].get("model"))
        path = ("step_motor_proof", "model", "mean_model")
        if _with_mean(policy, (), path, self._mean) != _with_mean(expected, (), path, self._mean):
            raise ValueError("owned proposal original preview integrity differs")
        result: dict[str, Any] = expected["step_motor_proof"]["model"]
        return result
