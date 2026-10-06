"""Fixed complete neural mean, owned seed envelopes; no execution authority."""

from pathlib import Path
from typing import Any, cast

import rosclaw.growth.frozen_payload_field as payload_module
from rosclaw.growth.canonical_json_snapshot import CanonicalJSONSnapshot
from rosclaw.growth.frozen_payload_field import FrozenPayloadField

from rosclaw_soccer.rsi.recurrent_sampling_motor import make_preview, make_sampling_view
from rosclaw_soccer.sim.contracts import hash_bytes


class RecurrentSamplingViewBuilder:
    """Validate once, retain WHOLE mean bytes, construct ordinary seed views.

    Persistence optimization only. A caller must publish the complete payload,
    use the normal sampling reader, and independently qualify native transport
    before collecting learning data. No decoder, simulator, factory callback,
    file-writing capability or physical evidence is provided here.
    """

    def __init__(self, mean_model: dict[str, Any]) -> None:
        paths = list(Path(__file__).parent.glob("*.py"))
        paths += list(Path(payload_module.__file__).parent.glob("*.py"))
        paths += [Path(__file__).parents[1] / "sim/contracts.py"]
        self._pins = {str(p): hash_bytes(p.read_bytes()) for p in paths}
        view = make_sampling_view(mean_model, seed=0)
        make_preview(view)  # Validate the owned copy, not only caller's input.
        self._cache = FrozenPayloadField(view["mean_model"])
        self._payload_hash: str = self._cache.payload_hash
        # Public cached hashing API also verifies that retained immutable bytes
        # have not changed without parsing the entire mean on every seed.
        self._root_hash = self._cache.document_hash({}, "mean_model")
        self._fields = CanonicalJSONSnapshot(
            {k: v for k, v in view.items() if k not in ("mean_model", "model_hash")}
        )
        self._mean_hash = str(view["mean_model"]["model_hash"])
        self._stable()

    def _stable(self) -> None:
        if (
            any(hash_bytes(Path(p).read_bytes()) != h for p, h in self._pins.items())
            or self._cache.payload_hash != self._payload_hash
            or self._cache.document_hash({}, "mean_model") != self._root_hash
        ):
            raise ValueError("complete fixed neural mean or sampling source changed")
        self._fields.verify()

    @property
    def payload_hash(self) -> str:
        self._stable()
        return self._payload_hash

    def envelope(self, *, seed: int) -> dict[str, Any]:
        if type(seed) is not int or not 0 <= seed < 2**32:
            raise ValueError("bounded integer independent exploration seed required")
        self._stable()
        fields = self._fields.restore()
        fields["seed"] = seed
        fields["model_hash"] = self._cache.document_hash(fields, "mean_model")
        return cast(dict[str, Any], self._cache.envelope(fields, "mean_model"))

    def mean_model(self) -> dict[str, Any]:
        """Return freshly owned COMPLETE mean for one normal payload write."""
        result: dict[str, Any] = self._cache.restore(self.envelope(seed=0), "mean_model")[
            "mean_model"
        ]
        return result

    def contract(self) -> dict[str, Any]:
        self._stable()
        return dict(
            schema="soccer.rsi.private_recurrent_sampling_view_builder.v1",
            payload_hash=self._payload_hash,
            actual_mean_model_hash=self._mean_hash,
            source_pins=dict(self._pins),
            complete_original_model_validation_at_allocation=True,
            all_logical_mean_fields_retained=True,
            native_transport_parity_requires_external_evidence=True,
            physics_or_learning_claimed=False,
            activation_ceiling="SIM_ONLY",
            promotion_authorized=False,
            hardware_authorized=False,
        )
