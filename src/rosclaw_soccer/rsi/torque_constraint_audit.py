"""Read-only numeric audit of actually applied simulated torques.

Parent fallback may bypass candidate-only rate/power limits. Non-parent
commands may not. This module has no policy, transport or simulator execution.
"""

import numpy as np


def audit_applied_torques(
    applied: np.ndarray,
    parent: np.ndarray,
    before_velocity: np.ndarray,
    hard_limits: np.ndarray,
    *,
    guard_scale: float,
    parent_deviation_ratio: float,
    delta_ratio: float,
    maximum_power_w: float,
) -> dict:
    applied, parent, velocity, hard = (
        np.asarray(value, dtype=np.float64)
        for value in (applied, parent, before_velocity, hard_limits)
    )
    scalars = [guard_scale, parent_deviation_ratio, delta_ratio, maximum_power_w]
    if (
        applied.ndim != 2
        or not len(applied)
        or parent.shape != applied.shape
        or velocity.shape != applied.shape
        or hard.shape != (applied.shape[1],)
        or not all(np.isfinite(value).all() for value in (applied, parent, velocity, hard))
        or not np.isfinite(scalars).all()
        or np.any(hard <= 0)
        or not 0 < guard_scale <= 0.85
        or not 0 <= parent_deviation_ratio <= 0.25
        or not 0 < delta_ratio <= 1
        or maximum_power_w <= 0
    ):
        raise ValueError("aligned finite actual torque, parent, before-state and limits required")
    # An exactly parent-valued command is observationally indistinguishable
    # from fallback; do not claim that its candidate-only guards passed.
    changed = np.any(applied != parent, axis=1)
    previous = np.vstack((np.zeros_like(applied[0]), applied[:-1]))
    deviation = np.abs(applied - parent) / hard
    rate = np.abs(applied - previous) / hard
    power = np.sum(np.abs(applied * velocity), axis=1)
    amplitude = np.abs(applied) / hard
    return dict(
        steps=len(applied),
        nonparent_steps=int(changed.sum()),
        parent_valued_steps=int((~changed).sum()),
        total_amplitude_violations=int(np.count_nonzero(amplitude > 0.85 + 1e-10)),
        parent_deviation_violations=int(
            np.count_nonzero(deviation > parent_deviation_ratio + 1e-10)
        ),
        nonparent_amplitude_violations=int(
            np.count_nonzero(amplitude[changed] > guard_scale + 1e-10)
        ),
        nonparent_rate_violations=int(np.count_nonzero(rate[changed] > delta_ratio + 1e-10)),
        nonparent_power_violations=int(np.count_nonzero(power[changed] > maximum_power_w + 1e-7)),
        maximum_parent_deviation_ratio=float(deviation.max()),
        maximum_nonparent_rate_ratio=float(rate[changed].max()) if changed.any() else 0.0,
        maximum_nonparent_power_w=float(power[changed].max()) if changed.any() else 0.0,
        maximum_parent_valued_power_w=float(power[~changed].max()) if (~changed).any() else 0.0,
        parent_valued_commands_not_candidate_guard_certified=True,
        independent_neural_replay=False,
    )
