"""Descriptive native contact diagnostics, separate from the frozen H300 exam."""

from typing import Any

import numpy as np


def contact_authority_metrics(arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    n = len(arrays["commands"])
    keys = {
        "commands": (n, 8, 3),
        "ball_forces": (n * 10, 8, 3),
        "nearest_foot_relative_speed_mps": (n, 8),
        "foot_gap_after_m": (n, 8),
        "teacher_ball_position_after": (n, 3),
        "teacher_applied_delta": (n, 29),
        "teacher_foot_velocity": (n, 3),
        "teacher_ball_velocity": (n, 3),
        "teacher_relative_velocity_after": (n, 3),
        "teacher_active": (n,),
        "teacher_fallback": (n,),
    }
    if (
        not 50 < n <= 600
        or any(
            arrays[k].shape != shape or not np.isfinite(arrays[k]).all()
            for k, shape in keys.items()
        )
        or arrays["teacher_active"].dtype != np.bool_
        or arrays["teacher_fallback"].dtype != np.bool_
    ):
        raise ValueError("aligned finite complete native contact-authority observations required")
    # Red finisher index3 is declared by this protocol, not inferred from labels.
    force = arrays["ball_forces"].reshape(n, 10, 8, 3)
    foot = force[:, :, 3, :2].max(axis=(1, 2)) > 1
    nonfoot = force[:, :, 3, 2].max(axis=1) > 1
    contact_rows = np.flatnonzero(foot[50:]) + 50
    first = int(contact_rows[0]) if len(contact_rows) else None
    observed = first is not None and first + 60 < n
    displacement = (
        arrays["teacher_ball_position_after"][first + 60]
        - arrays["teacher_ball_position_after"][first]
        if first is not None and first + 60 < n
        else None
    )
    age = 999
    close = 0
    for i in range(50, n):
        age = 0 if foot[i] else age + 1
        close += int(
            age <= 25
            and arrays["foot_gap_after_m"][i, 3] < 0.35
            and arrays["nearest_foot_relative_speed_mps"][i, 3] < 0.5
        )
    active = arrays["teacher_active"]
    before_speed = np.linalg.norm(
        arrays["teacher_foot_velocity"] - arrays["teacher_ball_velocity"], axis=1
    )
    after_speed = np.linalg.norm(arrays["teacher_relative_velocity_after"], axis=1)
    delta = arrays["teacher_applied_delta"]
    slew = np.diff(np.concatenate((np.zeros((1, 29)), delta)), axis=0)
    nonfallback = ~arrays["teacher_fallback"]
    return dict(
        controls=n,
        actual_foot_contact_controls=int(foot.sum()),
        actual_postwarmup_foot_contact_controls=int(foot[50:].sum()),
        actual_nonfoot_contact_controls=int(nonfoot.sum()),
        initial_contact_prefix=bool(foot[:50].any() or nonfoot[:50].any()),
        first_postwarmup_foot_contact_control=first,
        diagnostic_followup60_observed=bool(observed),
        diagnostic_followup_status="NO_POSTWARMUP_FOOT_CONTACT"
        if first is None
        else "OBSERVED"
        if observed
        else "RIGHT_CENSORED",
        diagnostic_ball_displacement60_m=None if displacement is None else displacement.tolist(),
        # These are not the old formal H300 deadline or a high-quality pass.
        diagnostic_followup_is_not_formal_H300_result=True,
        close_recent_comotion_controls=close,
        close_recent_comotion_is_not_possession_qualification=True,
        teacher_active_controls=int(active.sum()),
        teacher_nonzero_correction_controls=int(np.any(delta != 0, axis=1).sum()),
        teacher_fallback_controls=int(arrays["teacher_fallback"].sum()),
        maximum_applied_joint_correction_rad=float(np.max(abs(delta))),
        maximum_nonfallback_correction_slew_rad=float(np.max(abs(slew[nonfallback])))
        if nonfallback.any()
        else None,
        active_point_relative_speed_before_mean_mps=float(before_speed[active].mean())
        if active.any()
        else None,
        active_point_relative_speed_after_mean_mps=float(after_speed[active].mean())
        if active.any()
        else None,
        within_rollout_speed_difference_is_not_a_causal_gain_estimate=True,
        physical_possession_qualified=False,
        official_exam_pass_proven=False,
    )
