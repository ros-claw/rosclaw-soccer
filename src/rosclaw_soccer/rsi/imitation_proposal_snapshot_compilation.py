"""Owned imitation law after complete validation, without ancestor constructors.

Every call deep-owns and validates the entire model through the original
public preview. No validation flag/cache is accepted. Numeric allocation
reproduces the existing bounded snapshot; native parity remains external.
"""

import copy
from typing import Any

import numpy as np
from rosclaw.growth.bounded_query_anchor_guard import (
    BoundedQueryAnchorGuard,
    BoundedQueryDomainAnchorGuard,
)
from rosclaw.growth.indexed_anchor_output_memory import IndexedAnchorOutputMemory

from rosclaw_soccer.rsi.compiled_step_inference import CompiledStepMotor
from rosclaw_soccer.rsi.imitation_proposal_motor import CompiledImitationProposalMotor, make_preview
from rosclaw_soccer.rsi.output_memory_step_motor import CompiledOutputMemoryMotor
from rosclaw_soccer.rsi.output_memory_step_motor import make_preview as output_preview
from rosclaw_soccer.rsi.proposal_memory_motor import CompiledProposalMemoryMotor
from rosclaw_soccer.rsi.step_motor_execution import make_preview as warm_preview
from rosclaw_soccer.rsi.step_motor_phase_context import ContactPhaseMemory


def _array(value: Any) -> np.ndarray[Any, Any]:
    result = np.array(value, dtype=np.float64, copy=True)
    result.flags.writeable = False
    return result


def _layers(values: Any) -> list[tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]]:
    return [(_array(v["weight"]), _array(v["bias"])) for v in values]


def compile_imitation_snapshot(
    model: dict[str, Any],
) -> tuple[dict[str, Any], CompiledImitationProposalMotor]:
    """Validate a new owned model before allocating any executable numeric law."""
    owned = copy.deepcopy(model)
    policy = make_preview(owned)
    proposal = owned["frozen_parent"]
    initial = proposal["initial_actor"]
    smooth = initial["baseline"]["base_model"]
    nn = smooth["frozen_parent"]
    kernel = nn["frozen_parent"]
    base = kernel["encoder"]["base_model"]
    # Retain the exact output-memory parent's private commitment. This checks
    # that ancestor through its original validator, but does not repeatedly
    # construct the complete proposal graph at each inheritance level.
    parent_policy = output_preview(nn)
    parent = CompiledOutputMemoryMotor.__new__(CompiledOutputMemoryMotor)
    parent._guard = BoundedQueryAnchorGuard.from_dict(kernel["anchor_guard"])
    parent._warm = CompiledStepMotor.from_legacy_preview(warm_preview(base))
    parent._mean, parent._scale = _array(base["mean"]), _array(base["scale"])
    parent._layers = _layers(base["actor"]["layers"][:-1])
    random = kernel["encoder"]["frozen_random_features"]
    parent._random = (_array(random["weight"]), _array(random["bias"]))
    parent._head = _array(kernel["actor_readout"])
    parent._output_memory = IndexedAnchorOutputMemory.from_dict(nn["output_memory"])
    parent._output_memory._guard = BoundedQueryAnchorGuard.from_dict(
        parent._output_memory._guard.to_dict()
    )
    parent._encoder_hash = parent._output_memory.encoder_hash
    parent._residual_layers = _layers(nn["residual_layers"])
    parent._sampling = None
    parent._memory = ContactPhaseMemory()
    parent._policy_hash = parent_policy["policy_hash"]

    decoder = CompiledProposalMemoryMotor.__new__(CompiledProposalMemoryMotor)
    decoder._parent = parent
    decoder._layers = _layers(owned["residual_layers"])
    decoder._zero = not np.any(decoder._layers[-1][0]) and not np.any(decoder._layers[-1][1])
    decoder._cap = 0.2
    bank = proposal["protected_domain_bank"]
    decoder._guard = (
        BoundedQueryDomainAnchorGuard(bank["bank"], bandwidth=1e-4)
        if bank is not None
        else BoundedQueryAnchorGuard(initial["baseline"]["memory"]["observations"], bandwidth=1e-4)
    )
    decoder._sampling, decoder._noise = None, None
    decoder._memory = ContactPhaseMemory()
    decoder._policy_hash = policy["policy_hash"]

    result = CompiledImitationProposalMotor.__new__(CompiledImitationProposalMotor)
    result._decoder, result._parent = decoder, parent
    result._layers, result._memory = decoder._layers, decoder._memory
    result._guard, result._cap = decoder._guard, decoder._cap
    result._zero, result._policy_hash = decoder._zero, decoder._policy_hash
    return policy, result
