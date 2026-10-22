"""Two-body delivery/receiving outcome, separate from single-body reachability.

Only pre-receiver-contact ball motion within the original300 controls counts
for A precision. Subsequent B dribbling and the recovery tail cannot improve
that score. Functional legal-body receiving is NOT strict clean-foot capture.
"""

from typing import Any

import numpy as np

from rosclaw_soccer.rsi.neural_relay_incoming import incoming_delivery


def pass_transfer_outcome(
    ball_position: np.ndarray,
    ball_velocity: np.ndarray,
    forces: np.ndarray,
    goal: np.ndarray,
    *,
    whole_body_healthy: bool,
    passer_recovery_observed: bool,
    receiver_first_ready_control: int | None,
    receiver_excluded_contact: bool,
) -> dict[str, Any]:
    n = len(ball_position)
    if (
        ball_velocity.shape != (n, 3)
        or not np.isfinite(ball_velocity).all()
        or any(
            type(v) is not bool
            for v in (whole_body_healthy, passer_recovery_observed, receiver_excluded_contact)
        )
        or (
            receiver_first_ready_control is not None
            and (
                type(receiver_first_ready_control) is not int
                or not 15 <= receiver_first_ready_control <= n // 10
            )
        )
        or n % 10
    ):
        raise ValueError(
            "actual native velocity, causal receiving control and full recovery required"
        )
    formal = min(n, 3000)
    incoming = incoming_delivery(ball_position[:formal], forces[:formal], goal)
    first = incoming["A_first_foot_strike_substep"]
    end = incoming["incoming_window_end_exclusive"]
    closest = speed = None
    if incoming["incoming_error_observed"]:
        closest = first + int(np.argmin(np.linalg.norm(ball_position[first:end] - goal, axis=1)))
        speed = float(np.linalg.norm(ball_velocity[closest]))
    clean = not bool((forces[:, 0, 2] > 1).any())
    delivery = bool(
        whole_body_healthy
        and passer_recovery_observed
        and clean
        and incoming["incoming_error_observed"]
        and incoming["incoming_target_error_3d_before_B_contact_m"] <= 0.1
        and speed is not None
        and speed <= 3.0
    )
    received = bool(
        receiver_first_ready_control is not None
        and receiver_first_ready_control <= 300
        and not receiver_excluded_contact
        and whole_body_healthy
    )
    return dict(
        incoming=incoming,
        formal_native_steps=formal,
        incoming_closest_substep=closest,
        incoming_arrival_speed_mps=speed,
        passer_clean_foot=clean,
        whole_body_healthy=whole_body_healthy,
        passer_full100NNrecovery=passer_recovery_observed,
        precise_moderate_delivery=delivery,
        B_first_functional_controlled_receive=receiver_first_ready_control,
        B_legal_functional_receive_before_deadline=received,
        B_excluded_contact=receiver_excluded_contact,
        combined_precision_and_functional_receive=bool(delivery and received),
        strict_clean_foot_capture_regraded=False,
        team_game_verified=False,
    )
