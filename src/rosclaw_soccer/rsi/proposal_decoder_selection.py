"""Explicit numeric-constructor selection; no motion or approval interface."""

from pathlib import Path
from typing import Any

from rosclaw_soccer.rsi import proposal_snapshot_compilation as snapshot_module
from rosclaw_soccer.rsi.proposal_memory_motor import CompiledProposalMemoryMotor
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json

IMPLEMENTATIONS = ("reference", "owned_snapshot", "bounded_snapshot")


def compilation_contract(implementation: str) -> dict[str, Any] | None:
    if type(implementation) is not str or implementation not in IMPLEMENTATIONS:
        raise ValueError("explicit known proposal numeric constructor required")
    if implementation == "reference":
        return None
    result = dict(
        schema="soccer.rsi.offline_proposal_snapshot_compilation.v1",
        implementation="owned_snapshot",
        compiler_source_hash=hash_bytes(Path(snapshot_module.__file__).read_bytes()),
        selection_source_hash=hash_bytes(Path(__file__).read_bytes()),
        complete_owned_preview_validation=True,
        actor_weights_changed=False,
        physical_dynamics_changed=False,
        promotion_authorized=False,
        hardware_authorized=False,
    )
    if implementation == "bounded_snapshot":
        from rosclaw.growth import bounded_query_anchor_guard

        from rosclaw_soccer.rsi import bounded_query_proposal_compilation

        result.update(
            implementation=implementation,
            compiler_source_hash=hash_bytes(
                Path(bounded_query_proposal_compilation.__file__).read_bytes()
            ),
            original_snapshot_compiler_source_hash=hash_bytes(
                Path(snapshot_module.__file__).read_bytes()
            ),
            core_query_source_hash=hash_bytes(
                Path(bounded_query_anchor_guard.__file__).read_bytes()
            ),
            complete_logical_anchor_banks_retained=True,
            physical_parity_requires_separate_evidence=True,
        )
    return result


def validate_compilation_contract(value: Any) -> None:
    if type(value) is not dict or value.get("implementation") not in IMPLEMENTATIONS[1:]:
        raise ValueError("complete sealed owned proposal compilation contract required")
    expected = compilation_contract(value["implementation"])
    if hash_json(value) != hash_json(expected):
        raise ValueError("complete sealed owned proposal compilation contract required")


def select_proposal_decoder(
    policy: dict[str, Any], *, implementation: str
) -> CompiledProposalMemoryMotor:
    compilation_contract(implementation)
    if implementation == "reference":
        return CompiledProposalMemoryMotor(policy)
    if implementation == "bounded_snapshot":
        from rosclaw_soccer.rsi.bounded_query_proposal_compilation import (
            compile_bounded_query_proposal,
        )

        return compile_bounded_query_proposal(policy)
    return snapshot_module.compile_proposal_snapshot(policy)
