"""SIM-only provisional receiver credit, distinct from the official exam."""

import numpy as np


def receiver_continuation_credit(
    ball_forces,
    qvel,
    body_failed,
    *,
    controls,
    handoff_control,
    ball_velocity_index,
    return_direction,
    excluded_contact_observed,
    gamma=0.99,
):
    """Attribute observed events to B's sampled controls, including a discounted tail.

    Handoff is a readiness proxy, not proof of possession; force channels do
    not imply legality. `excluded_contact_observed` has no precise timestamp.
    """
    f, v, failed, direction = map(np.asarray, (ball_forces, qvel, body_failed, return_direction))
    if (
        f.ndim != 4
        or f.shape[1:] != (10, 2, 3)
        or not len(f)
        or v.ndim != 3
        or v.shape[:2] != (len(f), 10)
        or failed.dtype != np.bool_
        or failed.shape != (len(f), 2)
        or type(controls) is not int
        or not 1 <= controls <= len(f)
        or type(ball_velocity_index) is not int
        or not 0 <= ball_velocity_index <= v.shape[2] - 2
        or direction.shape != (2,)
        or not np.isfinite(direction).all()
        or abs(float(np.linalg.norm(direction)) - 1.0) > 1e-8
        or not np.isfinite(f).all()
        or (f < 0).any()
        or not np.isfinite(v).all()
        or type(excluded_contact_observed) is not bool
        or not np.isfinite(gamma)
        or not 0 < gamma <= 1
        or (
            handoff_control is not None
            and (type(handoff_control) is not int or not 0 <= handoff_control < len(f))
        )
    ):
        raise ValueError(
            "finite complete observed trajectory and causal event coordinates required"
        )
    flat = f.reshape(-1, 2, 3)
    velocities = v.reshape(-1, v.shape[2])
    feet = np.flatnonzero((flat[:, 1, :2] > 10).any(axis=1))
    others = np.flatnonzero(flat[:, 1, 2] > 1)
    failures = np.flatnonzero(failed.any(axis=1))
    foot = int(feet[0]) if feet.size else None
    other = int(others[0]) if others.size else None
    failure = int(failures[0]) if failures.size else None
    if failure is not None and failure != len(f) - 1:
        raise ValueError("body failure must terminate the observed physical trajectory")
    reward = np.zeros(controls, np.float32)
    events = {}

    def add(name, frame, value):
        assigned = min(frame, controls - 1)
        delay = frame - assigned
        amount = float(value * gamma**delay)
        reward[assigned] += np.float32(amount)
        events[name] = dict(
            observed_control=frame,
            assigned_control=assigned,
            delay=delay,
            undiscounted_value=value,
            discounted_value=amount,
        )

    if foot is not None and (other is None or foot < other):
        add("first_foot", foot // 10, 0.2)
    if other is not None:
        add("first_nonfoot", other // 10, -0.4)
    foot_qualified_handoff = (
        handoff_control is not None and foot is not None and foot // 10 < handoff_control
    )
    if foot_qualified_handoff:
        add("foot_qualified_handoff_proxy", handoff_control, 1.0)
        direction_speed = velocities[:, ball_velocity_index : ball_velocity_index + 2] @ direction
        strike = np.flatnonzero(
            ((flat[:, 1, :2] > 10).any(axis=1))
            & (direction_speed >= 1.0)
            & (np.arange(len(flat)) >= handoff_control * 10)
        )
        if strike.size:
            add("directed_return_proxy", int(strike[0]) // 10, 1.0)
    if failure is not None:
        add("body_failure", failure, -2.0)
    if excluded_contact_observed:
        add("excluded_contact_terminal", len(f) - 1, -2.0)
    return reward, dict(
        events=events,
        first_foot_substep=foot,
        first_nonfoot_substep=other,
        first_failure_control=failure,
        foot_qualified_handoff=bool(foot_qualified_handoff),
        handoff_is_not_possession=True,
        directed_return_is_not_teammate_pass=True,
        nonfoot_is_not_legality=True,
        official_exam_unchanged=True,
        hardware_authorized=False,
    )
