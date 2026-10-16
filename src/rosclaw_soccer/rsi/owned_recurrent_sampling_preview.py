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
from rosclaw_soccer.rsi.owned_smooth_preview import _with_mean
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
        # Capture one seed scalar, then snapshot the WHOLE caller document.
        # Compare its complete canonical bytes with a private expected document;
        # no caller dictionary or object identity is grandfathered. Returned
        # values are restored from private bytes, not the mutable input.
        seed = wrapped["seed"]
        if type(seed) is not int or not 0 <= seed < 2**32:
            raise ValueError("bounded integer recurrent sampling seed required")
        if type(wrapped["mean_model"]) is not dict or type(wrapped["model_hash"]) is not str:
            raise ValueError("complete canonical recurrent actor/critic/parent and seal required")
        incoming = CanonicalJSONSnapshot(wrapped)
        fields = dict(static, seed=seed, mean_model=None)
        digest = (
            "sha256:"
            + hashlib.sha256(_with_mean(fields, (), ("mean_model",), self._mean._data)).hexdigest()
        )
        expected = dict(fields, model_hash=digest)
        if incoming._data != _with_mean(expected, (), ("mean_model",), self._mean._data):
            raise ValueError(
                "complete canonical recurrent actor/critic/parent, sampling law, source, "
                "authority or original model seal changed"
            )
        owned = dict(static, seed=seed, mean_model=self._mean.restore(), model_hash=digest)
        return self._assemble_owned_policy(owned, digest)

    def _assemble_owned_policy(self, owned: dict[str, Any], digest: str) -> dict[str, Any]:
        """Internal materialization; external documents must go through preview()."""
        policy: dict[str, Any] = self._template.restore()
        policy["step_motor_proof"]["model"] = owned
        policy["recurrent_sampling_motor_proof"]["sampling_model_hash"] = digest
        policy["policy_hash"] = (
            "sha256:"
            + hashlib.sha256(
                _with_mean(
                    policy, (), ("step_motor_proof", "model", "mean_model"), self._mean._data
                )
            ).hexdigest()
        )
        self._stable()
        return policy

    def preview_from_seed(self, *, seed: int) -> dict[str, Any]:
        """Generate the COMPLETE original preview from the private frozen mean.

        Only an integer seed crosses this boundary. No caller model, sampling
        law or authority document is accepted, so there is no external mean to
        serialize/copy/check a second time. Allocation's whole-model validation,
        private byte/source checks and original downstream bind audit remain.
        The returned ordinary dictionary is freshly owned, not a shared view.
        """
        owned = self.sampling_view(seed=seed)
        return self._assemble_owned_policy(owned, owned["model_hash"])

    def sampling_view(self, *, seed: int) -> dict[str, Any]:
        """Fresh COMPLETE seed view from the fully validated private mean.

        No changed sampling law or caller-supplied model is accepted here.
        The original whole view hash includes the full actor/critic/parent.
        """
        self._stable()
        if type(seed) is not int or not 0 <= seed < 2**32:
            raise ValueError("bounded integer recurrent sampling seed required")
        result: dict[str, Any] = self._static.restore()
        result.update(seed=seed, mean_model=self._mean.restore())
        result["model_hash"] = (
            "sha256:"
            + hashlib.sha256(_with_mean(result, (), ("mean_model",), self._mean._data)).hexdigest()
        )
        self._stable()
        return result

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
