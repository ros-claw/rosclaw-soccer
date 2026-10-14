"""Explicit direction eligibility intervention, not learning or admission.

The diagnostic removes only the historical initial-ball vx restriction. The
learned memory gate, foundation-only exclusion and all swing limits remain.
"""

from __future__ import annotations

import math


def native_swing_eligible(
    *, memory_gate: bool, initial_vx: float, foundation_only: bool, mode: str = "legacy"
) -> bool:
    if (
        type(memory_gate) is not bool
        or type(foundation_only) is not bool
        or type(initial_vx) is not float
        or not math.isfinite(initial_vx)
        or type(mode) is not str
        or mode
        not in ("legacy", "memory_gate_both_directions_diagnostic", "bounded_contact_diagnostic")
    ):
        raise ValueError("explicit finite SIM swing eligibility diagnostic required")
    return not foundation_only and (
        mode == "bounded_contact_diagnostic"
        or (memory_gate and (initial_vx < 0 or mode == "memory_gate_both_directions_diagnostic"))
    )
