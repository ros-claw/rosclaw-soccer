"""SIM software handoff arithmetic for two LIVE neural joint-target skills.

No simulator state, policy history, motor, transport or training is owned here.
Callers must infer both proposals on the actual body, execute one shared tick,
and encode the executed target back into each active policy's own action ABI.
A smooth scalar schedule alone does not prove smooth or stable body motion.
"""

import math

import numpy as np

from rosclaw_soccer.rsi.shared_native_motor_world import NativePDParameters


def quintic_weight(control: int, *, start: int, duration: int) -> float:
    """Zero at start, one at start+duration; no phase/body reset is implied."""
    if (
        any(type(value) is not int for value in (control, start, duration))
        or control < 0
        or start < 0
        or not 1 <= duration <= 100
    ):
        raise ValueError("nonnegative actual control and bounded1..100 duration required")
    t = min(1.0, max(0.0, (control - start) / duration))
    return t * t * t * (10.0 + t * (-15.0 + 6.0 * t))


def blend_live_targets(
    outgoing: np.ndarray,
    incoming: np.ndarray,
    outgoing_pd: NativePDParameters,
    incoming_pd: NativePDParameters,
    *,
    weight: float,
) -> tuple[np.ndarray, NativePDParameters]:
    if not math.isfinite(weight) or not 0 <= weight <= 1:
        raise ValueError("finite convex handoff weight required")
    values = [np.asarray(value) for value in (outgoing, incoming)]
    if any(v.shape != (29,) or v.dtype.kind != "f" or not np.isfinite(v).all() for v in values):
        raise ValueError("two finite live HG29 joint targets required, never repair them")
    old, new = outgoing_pd.owned_copy(), incoming_pd.owned_copy()
    if not np.array_equal(old.effort, new.effort):
        raise ValueError("handoff must retain exactly the same physical effort caps")
    if weight == 0:
        return values[0].astype(np.float64, copy=True), old
    if weight == 1:
        return values[1].astype(np.float64, copy=True), new
    target = (1.0 - weight) * values[0].astype(np.float64) + weight * values[1].astype(np.float64)
    pd = NativePDParameters(
        (1.0 - weight) * old.stiffness + weight * new.stiffness,
        (1.0 - weight) * old.damping + weight * new.damping,
        old.effort,
    ).owned_copy()
    return target, pd
