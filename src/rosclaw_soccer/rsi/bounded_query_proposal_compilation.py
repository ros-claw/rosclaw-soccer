"""Opt-in owned numeric proposal with bounded anchor queries.

Full validation and immutable numeric allocation remain in the original
snapshot compiler. Only three private, owned search objects are replaced;
logical banks, policy law, weights and policy identity are unchanged. This is
not a default native compiler or evidence of full physical parity.
"""

from typing import Any

from rosclaw.growth.bounded_query_anchor_guard import (
    BoundedQueryAnchorGuard,
    BoundedQueryDomainAnchorGuard,
)

from rosclaw_soccer.rsi.proposal_memory_motor import CompiledProposalMemoryMotor
from rosclaw_soccer.rsi.proposal_snapshot_compilation import compile_proposal_snapshot


def compile_bounded_query_proposal(policy: dict[str, Any]) -> CompiledProposalMemoryMotor:
    """Compile a new owned snapshot; never modify another decoder or source."""
    result = compile_proposal_snapshot(policy)
    parent = result._parent
    for target in (parent, parent._output_memory):
        target._guard = BoundedQueryAnchorGuard.from_dict(target._guard.to_dict())
    bank = policy["step_motor_proof"]["model"]["protected_domain_bank"]
    if bank is None:
        result._guard = BoundedQueryAnchorGuard.from_dict(result._guard.to_dict())
    else:
        result._guard = BoundedQueryDomainAnchorGuard(
            result._guard.bank(), bandwidth=result._guard.bandwidth
        )
    return result
