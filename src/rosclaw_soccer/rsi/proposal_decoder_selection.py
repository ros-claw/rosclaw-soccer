"""Explicit numeric-constructor selection; no motion or approval interface."""

from pathlib import Path
from typing import Any

from rosclaw_soccer.rsi import proposal_snapshot_compilation as snapshot_module
from rosclaw_soccer.rsi.proposal_memory_motor import CompiledProposalMemoryMotor
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json

IMPLEMENTATIONS = ("reference", "owned_snapshot")


def compilation_contract(implementation: str) -> dict[str, Any] | None:
    if type(implementation) is not str or implementation not in IMPLEMENTATIONS:
        raise ValueError("explicit known proposal numeric constructor required")
    if implementation == "reference":
        return None
    return dict(
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


def validate_compilation_contract(value: Any) -> None:
    expected = compilation_contract("owned_snapshot")
    if type(value) is not dict or hash_json(value) != hash_json(expected):
        raise ValueError("complete sealed owned proposal compilation contract required")


def select_proposal_decoder(
    policy: dict[str, Any], *, implementation: str
) -> CompiledProposalMemoryMotor:
    compilation_contract(implementation)
    if implementation == "reference":
        return CompiledProposalMemoryMotor(policy)
    return snapshot_module.compile_proposal_snapshot(policy)
