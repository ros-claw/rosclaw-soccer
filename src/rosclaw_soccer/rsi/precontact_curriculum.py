"""TRAIN-only observed-state curriculum selection, never fabricated actions.

Selecting a state before a future failure is valid curriculum construction,
not proof of success. No DEV, post-contact state, unstable state or synthetic
pose is accepted. Actual reset/inference/physics parity must qualify separately.
"""

import numpy as np

from rosclaw_soccer.rsi.mjlab_motion_curriculum import TRAIN_MOTION_NAMES


def select_precontact_frames(
    *,
    source: str,
    foot_force: np.ndarray,
    nonfoot_force: np.ndarray,
    ball_to_feet: np.ndarray,
    native_qpos: np.ndarray,
    expected_foot: int,
    pelvis_qadr: int,
) -> dict[str, object]:
    if source not in TRAIN_MOTION_NAMES:
        raise ValueError("curriculum gradients may use TRAIN8 only, never DEV")
    if (
        type(expected_foot) is not int
        or expected_foot not in (0, 1)
        or type(pelvis_qadr) is not int
        or pelvis_qadr != 7
    ):
        raise ValueError("explicit foot identity and qualified ball-first native layout required")
    if not isinstance(foot_force, np.ndarray) or foot_force.ndim != 3:
        raise ValueError("complete native control/substep history required")
    n = len(foot_force)
    for value, shape in (
        (foot_force, (n, 10, 2)),
        (nonfoot_force, (n, 10)),
        (ball_to_feet, (n, 2, 3)),
        (native_qpos, (n, 43)),
    ):
        if (
            not isinstance(value, np.ndarray)
            or value.shape != shape
            or value.dtype.kind != "f"
            or not np.isfinite(value).all()
        ):
            raise ValueError("finite aligned raw physical observations required")
    if not 101 <= n <= 300 or (foot_force < 0).any() or (nonfoot_force < 0).any():
        raise ValueError("bounded nonnegative full physical contact record required")
    if np.max(np.abs(np.linalg.norm(native_qpos[:, 10:14], axis=1) - 1)) > 1e-5:
        raise ValueError("actual unit pelvis orientations required, no normalization repair")
    contact = ((foot_force > 1.0).any(axis=2) | (nonfoot_force > 1.0)).reshape(-1)
    hits = np.flatnonzero(contact)
    if len(hits):
        anchor = int(hits[0] // 10)
        reason = "BEFORE_FIRST_ROBOT_BALL_CONTACT"
    else:
        # A negative near-miss remains a negative; no success label is invented.
        eligible = np.arange(40, n - 60)
        distances = np.linalg.norm(ball_to_feet[eligible, expected_foot], axis=1)
        anchor = int(eligible[int(distances.argmin())])
        reason = "BEFORE_OBSERVED_NEAR_MISS"
    cases = []
    rejected = []
    for lead in (10, 20, 40, 80):
        frame = anchor - lead
        if frame < 0:
            rejected.append(dict(lead_controls=lead, reason="BEFORE_EPISODE_START"))
            continue
        if contact[: frame * 10].any():
            raise ValueError("post-contact history cannot be silently reset away")
        q = native_qpos[frame]
        upright = float(1 - 2 * (q[pelvis_qadr + 4] ** 2 + q[pelvis_qadr + 5] ** 2))
        if q[pelvis_qadr + 2] < 0.65 or upright < 0.9:
            rejected.append(dict(lead_controls=lead, reason="UNSTABLE_MEASURED_START", frame=frame))
            continue
        cases.append(
            dict(
                frame=frame,
                lead_controls=lead,
                reason=reason,
                measured_height_m=float(q[pelvis_qadr + 2]),
                measured_upright=upright,
                historical_contact_free=True,
                expert_success_label=False,
            )
        )
    return dict(
        source=source,
        anchor_frame=anchor,
        reason=reason,
        cases=cases,
        rejected=rejected,
        first_robot_contact_substep=int(hits[0]) if len(hits) else None,
        policy_improvement_proven=False,
    )
