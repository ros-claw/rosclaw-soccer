"""Private exact recurrent previews; not a decoder or physical qualification.

Validate the WHOLE actor, critic, historical parent and original preview at
allocation. Subsequent seed views still compare all canonical mean bytes and
all original sampling fields. The reference implementation is unchanged.
No running collector selects this optional compiler automatically.
"""

import copy
import hashlib
from pathlib import Path
from typing import Any

import rosclaw.growth.canonical_json_snapshot as snapshot_module
from rosclaw.growth.canonical_json_snapshot import CanonicalJSONSnapshot

from rosclaw_soccer.rsi import recurrent_sampling_motor
from rosclaw_soccer.rsi.owned_smooth_preview import _json, _with_mean
from rosclaw_soccer.sim.contracts import hash_bytes


class OwnedRecurrentSamplingPreview:
    """One immutable fully validated mean; freshly owned ordinary previews.

    This optimizes repeated validation, not physics or policy learning. Native
    transport/replay parity remains an external prerequisite for collection.
    The returned object contains the entire original model, including critic.
    """

    def __init__(self, mean_model: dict[str, Any]) -> None:
        if type(mean_model) is not dict:
            raise ValueError("complete ordinary recurrent mean dictionary required")
        paths = list(Path(__file__).parent.glob("*.py"))
        paths += list(Path(snapshot_module.__file__).parent.glob("*.py"))
        paths += [Path(__file__).parents[1] / "sim/contracts.py"]
        self._pins = {str(p): hash_bytes(p.read_bytes()) for p in paths}
        original = recurrent_sampling_motor.make_preview(
            recurrent_sampling_motor.make_sampling_view(copy.deepcopy(mean_model), seed=0)
        )
        view = original["step_motor_proof"]["model"]
        self._mean = CanonicalJSONSnapshot(view["mean_model"])
        self._mean_hash = self._mean.content_hash
        self._static = CanonicalJSONSnapshot(
            {k: v for k, v in view.items() if k not in ("mean_model", "seed", "model_hash")}
        )
        template = copy.deepcopy({k: v for k, v in original.items() if k != "policy_hash"})
        template["step_motor_proof"].pop("model")
        template["recurrent_sampling_motor_proof"].pop("sampling_model_hash")
        self._template = CanonicalJSONSnapshot(template)
        self._static_hash = self._static.content_hash
        self._template_hash = self._template.content_hash
        self._stable()

    def _stable(self) -> None:
        if (
            any(hash_bytes(Path(p).read_bytes()) != h for p, h in self._pins.items())
            or self._mean.verify() != self._mean_hash
            or self._static.verify() != self._static_hash
            or self._template.verify() != self._template_hash
        ):
            raise ValueError("fixed complete recurrent mean, template or dependency changed")

    def preview(self, wrapped: dict[str, Any]) -> dict[str, Any]:
        self._stable()
        static = self._static.restore()
        if type(wrapped) is not dict or set(wrapped) != set(static) | {
            "mean_model",
            "seed",
            "model_hash",
        }:
            raise ValueError("complete exact recurrent sampling document required")
        # Own before validating: caller mutation must not change the return.
        owned = CanonicalJSONSnapshot(wrapped).restore()
        if _json({k: owned[k] for k in static}) != _json(static):
            raise ValueError("fixed recurrent sampling law, source or authority changed")
        seed = owned["seed"]
        if type(seed) is not int or not 0 <= seed < 2**32:
            raise ValueError("bounded integer recurrent sampling seed required")
        mean_bytes = self._mean._data
        if type(owned["mean_model"]) is not dict or _json(owned["mean_model"]) != mean_bytes:
            raise ValueError("complete canonical recurrent actor/critic/parent changed")
        fields = {k: v for k, v in owned.items() if k != "model_hash"}
        digest = (
            "sha256:"
            + hashlib.sha256(_with_mean(fields, (), ("mean_model",), mean_bytes)).hexdigest()
        )
        if type(owned["model_hash"]) is not str or owned["model_hash"] != digest:
            raise ValueError("complete original recurrent sampling model seal changed")
        policy: dict[str, Any] = self._template.restore()
        policy["step_motor_proof"]["model"] = owned
        policy["recurrent_sampling_motor_proof"]["sampling_model_hash"] = digest
        policy["policy_hash"] = (
            "sha256:"
            + hashlib.sha256(
                _with_mean(policy, (), ("step_motor_proof", "model", "mean_model"), mean_bytes)
            ).hexdigest()
        )
        self._stable()
        return policy

    def validate_preview(self, policy: dict[str, Any]) -> dict[str, Any]:
        if type(policy) is not dict or type(policy.get("step_motor_proof")) is not dict:
            raise ValueError("complete ordinary recurrent sampling preview required")
        owned = CanonicalJSONSnapshot(policy).restore()
        expected = self.preview(owned["step_motor_proof"].get("model"))
        path = ("step_motor_proof", "model", "mean_model")
        if _with_mean(owned, (), path, self._mean._data) != _with_mean(
            expected, (), path, self._mean._data
        ):
            raise ValueError("complete original recurrent preview integrity changed")
        result: dict[str, Any] = expected["step_motor_proof"]["model"]
        return result
