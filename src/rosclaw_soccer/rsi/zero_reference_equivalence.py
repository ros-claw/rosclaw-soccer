"""Prove zero-reference NAV controls survive train-only normalization changes.

No physics or source authentication. Hidden states and nonzero students are
NOT equivalent. Original full physical receipts are still required.
"""

from collections.abc import Callable
from typing import Any, cast

import numpy as np
from rosclaw.growth.causal_residual_memory import initial_parameters

from rosclaw_soccer.rsi.reference_student_execution import own_model as _own_model
from rosclaw_soccer.sim.contracts import hash_json

own_model = cast(Callable[[dict[str, Any]], dict[str, Any]], _own_model)


def require_zero_reference_equivalence(
    recorded: dict[str, Any], current: dict[str, Any]
) -> dict[str, Any]:
    a, b = own_model(recorded), own_model(current)
    expected = initial_parameters(135, 12, hidden_dimension=64, seed=2026101902)
    if (
        a["parameters"] != expected
        or b["parameters"] != expected
        or np.any(np.asarray(expected["head_weight"]) != 0)
        or np.any(np.asarray(expected["head_bias"]) != 0)
        or any(
            a[k] != b[k]
            for k in ("world_hash", "parent_model_hash", "source_hashes", "execution_source_hash")
        )
    ):
        raise ValueError("exact original zero-head controls and unchanged execution ABI required")
    value = dict(
        schema="soccer.rsi.zero_reference_normalization_equivalence.v1",
        recorded_model_hash=a["model_hash"],
        current_model_hash=b["model_hash"],
        parameters_hash=hash_json(expected),
        normalization_may_differ=True,
        hidden_states_may_differ=True,
        normalized_inputs_are_finite_and_clipped=True,
        exact_zero_head_for_all_finite_hidden_states=True,
        zero_projection_preserves_navigation_bits=True,
        only_zero_reference_behavior_equivalent=True,
        nonzero_student_behavior_equivalent=False,
        original_physical_receipts_still_required=True,
        source_authentication_performed_here=False,
        new_physical_steps=0,
        optimizer_updates=0,
        hardware_authorized=False,
        promotion_authorized=False,
    )
    return dict(value, report_hash=hash_json(value))
