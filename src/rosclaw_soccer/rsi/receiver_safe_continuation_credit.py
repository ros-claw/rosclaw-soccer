"""Safety-gated offline continuation credit; never an online policy input."""

import numpy as np

from rosclaw_soccer.rsi.receiver_continuation_credit import receiver_continuation_credit


def receiver_safe_continuation_credit(
    ball_forces,
    qvel,
    body_failed,
    *,
    termination,
    controls,
    handoff_control,
    ball_velocity_index,
    return_direction,
    excluded_contact_observed,
    gamma=0.99,
):
    """Gate bonuses on fully observed safety; censored/failed runs earn no bonus."""
    f, failed = np.asarray(ball_forces), np.asarray(body_failed)
    old, evidence = receiver_continuation_credit(
        f,
        qvel,
        failed,
        controls=controls,
        handoff_control=handoff_control,
        ball_velocity_index=ball_velocity_index,
        return_direction=return_direction,
        excluded_contact_observed=excluded_contact_observed,
        gamma=gamma,
    )
    if termination == "WORLD_HORIZON_COMPLETE":
        if len(f) != 600 or failed.any():
            raise ValueError("complete horizon requires 600 controls and no body failure")
    elif termination == "NO_RECEIVING_HANDOFF_WITHIN_ORIGINAL_VELOCITY_H300":
        if len(f) != 300 or handoff_control is not None or failed.any():
            raise ValueError("H300 censor requires no handoff and no observed body failure")
    elif termination == "BODY_FAILURE":
        if not failed[-1].any():
            raise ValueError("body-failure termination requires observed final failure")
    else:
        raise ValueError("unsupported or ambiguous physical termination")
    eligible = termination == "WORLD_HORIZON_COMPLETE" and not excluded_contact_observed
    reward = np.zeros_like(old)
    events = {}
    for name, event in evidence["events"].items():
        accepted = event["undiscounted_value"] < 0 or eligible
        if accepted:
            reward[event["assigned_control"]] += np.float32(event["discounted_value"])
        events[name] = dict(event, accepted=accepted)
    return reward, dict(
        events=events,
        full_horizon_safety_eligible=eligible,
        safety_observation_censored=termination
        == "NO_RECEIVING_HANDOFF_WITHIN_ORIGINAL_VELOCITY_H300",
        observed_termination=termination,
        offline_future_gate_is_not_actor_input=True,
        handoff_is_not_possession=True,
        directed_return_is_not_teammate_pass=True,
        nonfoot_is_not_legality=True,
        official_exam_unchanged=True,
        hardware_authorized=False,
    )
