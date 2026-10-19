"""Opt-in storage compilation of legacy Core domain guards for SIM research.

Reuses Core's whole-bank validation, immutable JSON snapshot and unchanged delegated
gate math. No row/parent/critic eviction, weight change or hash-by-object-id
cache. Reversibility is checked on the WHOLE original numeric graph, including
aliases and mutable recurrent histories. Physics parity remains external.
"""

import copy
from typing import Any, cast

import numpy as np
from rosclaw.growth.bounded_query_anchor_guard import BoundedQueryDomainAnchorGuard
from rosclaw.growth.canonical_json_snapshot import CanonicalJSONSnapshot
from rosclaw.growth.domain_anchor_bank import DomainAnchorGuard, validate_domain_anchor_bank

from rosclaw_soccer.rsi.fixed_recurrent_reference_audit import _numeric_graph_hash
from rosclaw_soccer.rsi.recurrent_sampling_episode_factory import RecurrentSamplingEpisodeFactory
from rosclaw_soccer.rsi.recurrent_sampling_motor import CompiledRecurrentSamplingMotor
from rosclaw_soccer.sim.contracts import hash_json

GUARD_TYPES = {
    "DomainAnchorGuard": DomainAnchorGuard,
    "BoundedQueryDomainAnchorGuard": BoundedQueryDomainAnchorGuard,
}


class SnapshotDomainAnchorGuard:
    """Same numeric gate with a complete byte-backed serialization payload."""

    _anchors: np.ndarray
    dimension: int
    bandwidth: float
    _tree: Any
    bank_hash: str

    def __init__(self, original: DomainAnchorGuard) -> None:
        if type(original) not in GUARD_TYPES.values() or set(vars(original)) != {
            "_bank",
            "_anchors",
            "dimension",
            "bandwidth",
            "_tree",
            "bank_hash",
        }:
            raise ValueError("exact complete original legacy domain guard required")
        bank = original.bank()
        validate_domain_anchor_bank(bank)
        fields = copy.deepcopy({k: v for k, v in vars(original).items() if k != "_bank"})
        self.__dict__.update(fields)
        self._source_guard_kind = type(original).__name__
        self._bank_snapshot = CanonicalJSONSnapshot(bank)
        self._anchors.flags.writeable = False
        if self.bank() != bank or self.to_dict() != original.to_dict():
            raise ValueError("whole logical bank or numeric metric changed in storage compilation")

    def bank(self) -> dict[str, Any]:
        result: dict[str, Any] = self._bank_snapshot.restore()
        return result

    def gate(self, latent: Any) -> float:
        # Legacy Core's numeric methods use only the copied metric fields, not
        # the serialization payload. Delegation avoids copying optimizer/math.
        cls = GUARD_TYPES[self._source_guard_kind]
        result: float = cls.gate(cast(DomainAnchorGuard, self), latent)
        return result

    def gates(self, latents: Any) -> np.ndarray:
        result: np.ndarray = DomainAnchorGuard.gates(cast(DomainAnchorGuard, self), latents)
        return result

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = DomainAnchorGuard.to_dict(cast(DomainAnchorGuard, self))
        return result

    def original_storage(self) -> DomainAnchorGuard:
        """Fresh exact logical original, not a policy constructor shortcut."""
        restored = object.__new__(GUARD_TYPES[self._source_guard_kind])
        restored.__dict__.update(
            copy.deepcopy(
                {
                    k: v
                    for k, v in vars(self).items()
                    if k not in ("_bank_snapshot", "_source_guard_kind")
                }
            )
        )
        restored._bank = self.bank()
        return restored


def _guards(root: Any, cls: type | tuple[type, ...]) -> list[Any]:
    seen: set[int] = set()
    found = []

    def visit(value: Any, depth: int) -> None:
        if depth > 128 or len(seen) > 65536:
            raise ValueError("bounded already-authenticated numeric object graph required")
        if value is None or type(value) in (str, bool, int, float, bytes) or id(value) in seen:
            return
        seen.add(id(value))
        if type(value) in (cls if isinstance(cls, tuple) else (cls,)):
            found.append(value)
        elif type(value) is CanonicalJSONSnapshot:
            # Frozen slotted dataclass: complete bytes were authenticated by
            # the whole numeric hash, and cannot contain live runtime objects.
            value.verify()
        elif type(value) is dict:
            # The preceding full numeric hash authenticates complete JSON
            # payloads. They contain no runtime guard object to transform.
            if type(value.get("schema")) is not str:
                for item in value.values():
                    visit(item, depth + 1)
        elif type(value) in (list, tuple):
            for item in value:
                visit(item, depth + 1)
        elif type(value).__module__.startswith(("rosclaw.", "rosclaw_soccer.")):
            visit(vars(value), depth + 1)

    visit(root, 0)
    return found


def compile_snapshot_banks(
    decoder: CompiledRecurrentSamplingMotor,
) -> tuple[CompiledRecurrentSamplingMotor, dict[str, Any]]:
    if type(decoder) is not CompiledRecurrentSamplingMotor:
        raise ValueError("exact original recurrent sampling numeric decoder required")
    before = _numeric_graph_hash(decoder)
    originals = _guards(decoder, tuple(GUARD_TYPES.values()))
    if _guards(decoder, SnapshotDomainAnchorGuard):
        raise ValueError("explicit once-only original-to-snapshot compilation required")
    replacements = {id(g): SnapshotDomainAnchorGuard(g) for g in originals}
    compiled = copy.deepcopy(decoder, replacements)
    after = _numeric_graph_hash(compiled)
    snapshots = _guards(compiled, SnapshotDomainAnchorGuard)
    if len(snapshots) != len(originals):
        raise ValueError("every original domain guard and alias must be retained")
    reverse = {id(g): g.original_storage() for g in snapshots}
    restored = copy.deepcopy(compiled, reverse)
    if _numeric_graph_hash(restored) != before or _numeric_graph_hash(decoder) != before:
        raise ValueError("whole original numeric graph must be exactly reversible and untouched")
    review = dict(
        schema="soccer.rsi.snapshot_domain_storage_compilation.v1",
        original_numeric_graph_hash=before,
        compiled_numeric_graph_hash=after,
        reversible_whole_graph_and_aliases_verified=True,
        converted_guard_count=len(originals),
        source_guard_types=[type(g).__name__ for g in originals],
        complete_logical_banks=[g.bank_hash for g in snapshots],
        core_gate_math_overridden=False,
        actor_or_critic_weights_changed=False,
        protected_rows_evicted=False,
        mutable_state_shared_with_donor=False,
        physical_parity_verified=False,
        teacher_qualified=False,
        promotion_authorized=False,
        hardware_authorized=False,
    )
    review["report_hash"] = hash_json(review)
    return compiled, review


def restore_original_banks(
    decoder: CompiledRecurrentSamplingMotor,
) -> CompiledRecurrentSamplingMotor:
    """Diagnostic inverse at the CURRENT recurrent boundary, never a reset."""
    if type(decoder) is not CompiledRecurrentSamplingMotor:
        raise ValueError("exact compiled current numeric decoder required")
    _numeric_graph_hash(decoder)
    snapshots = _guards(decoder, SnapshotDomainAnchorGuard)
    restored: CompiledRecurrentSamplingMotor = copy.deepcopy(
        decoder, {id(g): g.original_storage() for g in snapshots}
    )
    _numeric_graph_hash(restored)
    return restored


class SnapshotRecurrentSamplingEpisodeFactory(RecurrentSamplingEpisodeFactory):
    """Original full constructor, then reversible private storage compilation.

    Not selected by any existing native CLI or collector. Complete ordinary policy
    documents and original model/source checks remain in the inherited bind path.
    """

    def __init__(self, mean_model: dict[str, Any]) -> None:
        super().__init__(mean_model)
        compiled, review = compile_snapshot_banks(self._prototype)
        self._prototype = compiled
        self._prototype_hash = _numeric_graph_hash(compiled)
        self._storage_review = CanonicalJSONSnapshot(review)
        self._stable()

    def contract(self) -> dict[str, Any]:
        original = super().contract()
        review = self._storage_review.restore()
        if review["compiled_numeric_graph_hash"] != self._prototype_hash:
            raise ValueError("fixed storage compilation review must bind the whole prototype")
        return dict(
            schema="soccer.rsi.snapshot_recurrent_sampling_factory.v1",
            original_factory_contract=original,
            storage_compilation=review,
            complete_original_constructor_executed=True,
            physical_parity_verified=False,
            automatically_selected_by_native_collector=False,
            promotion_authorized=False,
            hardware_authorized=False,
        )
