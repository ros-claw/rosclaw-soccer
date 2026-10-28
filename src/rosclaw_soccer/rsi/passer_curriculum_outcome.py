"""Strict SIM-only physical pass/receive observations, distinct from old handoff."""

import numpy as np


def receiver_first_foot_velocity_change(ball_forces: np.ndarray, ball_velocity: np.ndarray) -> dict:
    """Measure first receiver-foot event from sampled native substeps.

    The one-substep velocity change includes gravity and any simultaneous
    contacts; it is not an isolated foot impulse or contact-point velocity.
    """
    if (
        not isinstance(ball_forces, np.ndarray)
        or ball_forces.ndim != 4
        or ball_forces.shape[1:] != (10, 2, 3)
        or not len(ball_forces)
        or not np.isfinite(ball_forces).all()
        or (ball_forces < 0).any()
        or not isinstance(ball_velocity, np.ndarray)
        or ball_velocity.shape != (len(ball_forces), 10, 3)
        or not np.isfinite(ball_velocity).all()
    ):
        raise ValueError("aligned finite native force and ball velocity substeps required")
    forces = ball_forces.reshape(-1, 2, 3)
    velocities = ball_velocity.reshape(-1, 3)
    events = np.flatnonzero(np.max(forces[:, 1, :2], axis=1) > 10)
    if not len(events):
        return dict(
            first_foot_substep=None,
            pre_speed_mps=None,
            post_speed_mps=None,
            velocity_change_mps=None,
            concurrent_nonfoot=None,
            pre_sample_available=False,
            isolated_foot_impulse=False,
        )
    index = int(events[0])
    return dict(
        first_foot_substep=index,
        pre_speed_mps=float(np.linalg.norm(velocities[index - 1])) if index else None,
        post_speed_mps=float(np.linalg.norm(velocities[index])),
        velocity_change_mps=float(np.linalg.norm(velocities[index] - velocities[index - 1]))
        if index
        else None,
        concurrent_nonfoot=bool(forces[index, 1, 2] > 1),
        pre_sample_available=bool(index),
        isolated_foot_impulse=False,
    )


def delivery_center_speeds(
    ball_qvel: np.ndarray, *, passer_substep: int | None, receiver_substep: int | None
) -> dict:
    """Measure sampled ball-center speed after A and just before B's contact.

    These are world-frame center speeds, not surface-relative contact speeds.
    Substep velocities must come from the post-step native physics trace.
    """
    if (
        not isinstance(ball_qvel, np.ndarray)
        or ball_qvel.ndim != 2
        or ball_qvel.shape[1] != 3
        or len(ball_qvel) == 0
        or not np.isfinite(ball_qvel).all()
    ):
        raise ValueError("finite post-step ball-center velocities are required")
    for index in (passer_substep, receiver_substep):
        if index is not None and (type(index) is not int or not 0 <= index < len(ball_qvel)):
            raise ValueError("contact substep must index the velocity trace")
    after_a = None if passer_substep is None else float(np.linalg.norm(ball_qvel[passer_substep]))
    before_b = None
    if (
        passer_substep is not None
        and receiver_substep is not None
        and passer_substep < receiver_substep
    ):
        before_b = float(np.linalg.norm(ball_qvel[receiver_substep - 1]))
    return {
        "ball_center_speed_after_passer_mps": after_a,
        "ball_center_speed_before_receiver_mps": before_b,
        "speed_is_not_contact_point_relative_velocity": True,
    }


def strict_pass_receive(ball_forces: np.ndarray, body_failed: np.ndarray, *, handoff: bool) -> dict:
    """Report separate receiver-clean and both-body-clean native histories."""
    n = len(ball_forces)
    if (
        not isinstance(ball_forces, np.ndarray)
        or ball_forces.shape != (n, 10, 2, 3)
        or n < 1
        or not np.isfinite(ball_forces).all()
        or (ball_forces < 0).any()
        or not isinstance(body_failed, np.ndarray)
        or body_failed.shape != (n, 2)
        or body_failed.dtype != np.bool_
        or type(handoff) is not bool
    ):
        raise ValueError("complete finite native contact and two-body health history required")
    substeps = ball_forces.reshape(-1, 2, 3)
    passer = np.flatnonzero(np.max(substeps[:, 0, :2], axis=1) > 10)
    receiver = np.flatnonzero(np.max(substeps[:, 1, :2], axis=1) > 10)
    other = np.flatnonzero(substeps[:, 1, 2] > 1)
    passer_other = np.flatnonzero(substeps[:, 0, 2] > 1)
    first_a = int(passer[0]) if len(passer) else None
    first_b = int(receiver[0]) if len(receiver) else None
    before = first_a is not None and first_b is not None and first_a < first_b
    healthy = not bool(body_failed.any())
    clean = bool(before and not len(other) and healthy and handoff)
    return dict(
        actual_passer_foot_substep=first_a,
        actual_receiver_foot_substep=first_b,
        passer_foot_precedes_receiver=before,
        receiver_nonfoot_substep=int(other[0]) if len(other) else None,
        passer_nonfoot_substep=int(passer_other[0]) if len(passer_other) else None,
        both_bodies_healthy=healthy,
        original_handoff=handoff,
        strict_receiver_clean_handoff=clean,
        strict_both_clean_handoff=bool(clean and not len(passer_other)),
        passer_nonfoot_not_evaluated=False,
        goal_complete=False,
        hardware_authorized=False,
    )
