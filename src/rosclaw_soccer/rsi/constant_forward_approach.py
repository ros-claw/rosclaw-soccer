"""One-factor SIM approach diagnostic, not a trained motor policy.

Keep the original forward cap, lateral law, yaw, IK and residual. Only replace
the before-contact distance-proportional forward request with the same cap.
No source contact time, case/seed, future trajectory or outcome is an input.
"""

import math


def apply_constant_forward(
    requested: tuple[float, float, float],
    projected: tuple[float, float, float],
    *,
    before_contact: bool,
    cap: float,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    if (
        type(before_contact) is not bool
        or type(cap) is not float
        or cap != 1.4
        or any(type(a) is not tuple or len(a) != 3 for a in (requested, projected))
        or any(
            type(x) is not float or not math.isfinite(x) for a in (requested, projected) for x in a
        )
        or not 0.0 <= projected[0] <= cap
        or abs(projected[1]) > 0.2
        or projected[2] != 0.0
        or requested[2] != 0.0
    ):
        raise ValueError("unchanged1.4m/s cap and finite original navigation required")
    if not before_contact:
        return requested, projected
    return (cap, *requested[1:]), (cap, *projected[1:])
