"""Opt-in offline numeric compilation after a complete owned-policy validation.

No validation result is cached between calls. Historical constructors remain
unchanged. This module exposes no simulator, actuator, activation or approval.
Native adoption requires separate compiler provenance and replay evidence.
"""

import copy
from typing import Any

import numpy as np
from rosclaw.growth.anchor_kernel import AnchorKernelGuard
from rosclaw.growth.domain_anchor_bank import DomainAnchorGuard
from rosclaw.growth.indexed_anchor_output_memory import IndexedAnchorOutputMemory

from rosclaw_soccer.rsi.compiled_step_inference import CompiledStepMotor
from rosclaw_soccer.rsi.output_memory_step_motor import CompiledOutputMemoryMotor
from rosclaw_soccer.rsi.output_memory_step_motor import make_preview as output_preview
from rosclaw_soccer.rsi.proposal_memory_motor import CompiledProposalMemoryMotor, make_preview
from rosclaw_soccer.rsi.step_motor_execution import make_preview as warm_preview
from rosclaw_soccer.rsi.step_motor_phase_context import ContactPhaseMemory


def _array(value: Any) -> np.ndarray[Any, Any]:
    result = np.array(value, dtype=np.float64, copy=True)
    result.flags.writeable = False
    return result


def _layers(values: Any) -> list[tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]]:
    return [(_array(v["weight"]), _array(v["bias"])) for v in values]


def _allocate_output_decoder() -> CompiledOutputMemoryMotor:
    return CompiledOutputMemoryMotor.__new__(CompiledOutputMemoryMotor)


def compile_proposal_snapshot(policy: dict[str, Any]) -> CompiledProposalMemoryMotor:
    """Compile the exact existing law, without repeating ancestor constructors.

    The existing public preview validator recursively checks the complete model
    graph, receipts, logical banks, sources and authority flags before any
    executable numeric object is returned. The check runs on a deep-owned copy.
    Only explicit proposal policies are accepted; this is not a general bypass
    for legacy or hardware policies. All numeric views own read-only storage.
    """
    snapshot = copy.deepcopy(policy)
    model = snapshot["step_motor_proof"]["model"]
    if make_preview(model) != snapshot:
        raise ValueError("proposal snapshot lost complete sealed policy binding")
    initial = model["initial_actor"]
    smooth = initial["baseline"]["base_model"]
    nn = smooth["frozen_parent"]
    kernel = nn["frozen_parent"]
    base = kernel["encoder"]["base_model"]
    # Preserve the parent's exact private commitment too. This independently
    # validates the output-memory ancestor; it does not construct its ancestors
    # and revalidate the large recursive graph at every inheritance level.
    parent_policy = output_preview(nn)
    parent = _allocate_output_decoder()
    parent._guard = AnchorKernelGuard.from_dict(kernel["anchor_guard"])
    parent._warm = CompiledStepMotor.from_legacy_preview(warm_preview(base))
    parent._mean, parent._scale = _array(base["mean"]), _array(base["scale"])
    parent._layers = _layers(base["actor"]["layers"][:-1])
    random = kernel["encoder"]["frozen_random_features"]
    parent._random = (_array(random["weight"]), _array(random["bias"]))
    parent._head = _array(kernel["actor_readout"])
    parent._output_memory = IndexedAnchorOutputMemory.from_dict(nn["output_memory"])
    parent._encoder_hash = parent._output_memory.encoder_hash
    parent._residual_layers = _layers(nn["residual_layers"])
    parent._sampling = None
    parent._memory = ContactPhaseMemory()
    parent._policy_hash = parent_policy["policy_hash"]
    result = CompiledProposalMemoryMotor.__new__(CompiledProposalMemoryMotor)
    result._parent = parent
    result._layers = _layers(model["residual_layers"])
    result._zero = not np.any(result._layers[-1][0]) and not np.any(result._layers[-1][1])
    result._cap = 0.2
    result._guard = (
        DomainAnchorGuard(model["protected_domain_bank"]["bank"], bandwidth=1e-4)
        if model["protected_domain_bank"] is not None
        else AnchorKernelGuard(initial["baseline"]["memory"]["observations"], bandwidth=1e-4)
    )
    result._sampling = None
    result._noise = None
    result._memory = ContactPhaseMemory()
    result._policy_hash = snapshot["policy_hash"]
    return result
