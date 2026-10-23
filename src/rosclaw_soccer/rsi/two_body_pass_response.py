"""Actual pre-receiver-contact response, never a predicted success label.

All indices refer to native2ms samples. Receiver interception and the neural
standing tail cannot improve A's frozen delivery measurement retrospectively.
"""

from typing import Any

import numpy as np


def refinement_rank(outcome: dict[str, Any]) -> tuple[int, int, int, int, float]:
    """New optimizer cost, NOT a change to any old physical success label."""
    from rosclaw_soccer.rsi.pass_transfer_learning import controlled_pass_label

    controlled = controlled_pass_label(outcome)
    error = outcome["incoming"]["incoming_target_error_3d_before_B_contact_m"]
    speed = outcome["incoming_arrival_speed_mps"]
    strict = outcome["combined_precision_and_functional_receive"]
    if type(strict) is not bool or any(
        v is not None and (type(v) not in (int, float) or not np.isfinite(v) or v < 0)
        for v in (error, speed)
    ):
        raise ValueError("finite actual error/speed and typed unchanged strict result required")
    score = -1e6 if error is None or speed is None else -error / 0.1 - max(0.0, speed - 3.0) / 0.2
    return (
        int(
            outcome["whole_body_healthy"]
            and outcome["passer_full100NNrecovery"]
            and not outcome["B_excluded_contact"]
        ),
        int(outcome["passer_clean_foot"]),
        int(controlled),
        int(strict),
        float(score),
    )


def measured_response(
    positions: np.ndarray,
    velocities: np.ndarray,
    initial: np.ndarray,
    goal: np.ndarray,
    outcome: dict[str, Any],
) -> np.ndarray | None:
    n = len(positions)
    if (
        positions.shape != (n, 3)
        or velocities.shape != (n, 3)
        or n < 1
        or initial.shape != (3,)
        or goal.shape != (3,)
        or any(not np.isfinite(a).all() for a in (positions, velocities, initial, goal))
    ):
        raise ValueError("finite actual native ball position/velocity and initial/target required")
    incoming = outcome["incoming"]
    k = outcome["incoming_closest_substep"]
    if incoming["incoming_error_observed"] is False:
        if (
            k is not None
            or outcome["incoming_arrival_speed_mps"] is not None
            or incoming["incoming_target_error_3d_before_B_contact_m"] is not None
        ):
            raise ValueError("missing delivery cannot contain fabricated arrival measurements")
        return None
    first, end = incoming["A_first_foot_strike_substep"], incoming["incoming_window_end_exclusive"]
    formal = min(n, 3000)
    b = incoming["B_first_ball_contact_substep"]
    if (
        incoming["incoming_error_observed"] is not True
        or any(type(x) is not int for x in (first, end, k))
        or not 0 <= first <= k < end <= formal
        or outcome["formal_native_steps"] != formal
        or (b is not None and (type(b) is not int or b < 0))
        or end != min(formal, b if b is not None else formal)
    ):
        raise ValueError("exact actual pre-B-contact/formal-window arrival index required")
    distances = np.linalg.norm(positions[first:end] - goal, axis=1)
    if first + int(np.argmin(distances)) != k:
        raise ValueError("closest sample does not match actual original observation window")
    speed = float(np.linalg.norm(velocities[k]))
    error = float(np.linalg.norm(positions[k] - goal))
    # Scalar rederivation tolerance only, not a relaxation of ANY task threshold.
    if not np.isclose(
        speed, outcome["incoming_arrival_speed_mps"], rtol=0, atol=1e-12
    ) or not np.isclose(
        error, incoming["incoming_target_error_3d_before_B_contact_m"], rtol=0, atol=1e-12
    ):
        raise ValueError("actual raw arrival error/speed contradict original physical outcome")
    forward = goal[:2] - initial[:2]
    if np.linalg.norm(forward) < 0.5:
        raise ValueError("nondegenerate declared delivery ray required")
    forward = forward / np.linalg.norm(forward)
    lateral = np.array([-forward[1], forward[0]])
    delta = positions[k] - goal
    return np.array([delta[:2] @ lateral, delta[:2] @ forward, delta[2], speed - 2.8], np.float64)


def new_distinct_proposals(
    candidates: list[dict[str, Any]], executed: list[np.ndarray]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Do not spend more worlds on an identical already measured control.

    This deduplicates control parameters, never drops physical failure records
    or resamples/changes a declared initial ball state.
    """

    def key(z: np.ndarray) -> bytes:
        z = np.array(z, np.float64, copy=True)
        if z.shape != (2, 6) or not np.isfinite(z).all() or (abs(z) > 1).any():
            raise ValueError("finite bounded actual twelve-dimensional control required")
        z[z == 0] = 0.0
        return z.tobytes()

    old = {key(z) for z in executed}
    seen = set()
    fresh = []
    duplicate = []
    for i, c in enumerate(candidates):
        k = key(np.asarray(c["latent"]))
        if k in old or k in seen:
            duplicate.append(
                dict(
                    candidate_index=i,
                    reason="ALREADY_EXECUTED" if k in old else "DUPLICATE_PROPOSAL",
                )
            )
        else:
            fresh.append(dict(candidate_index=i, **c))
            seen.add(k)
    return fresh, duplicate
