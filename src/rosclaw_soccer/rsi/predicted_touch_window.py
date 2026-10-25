"""Score only observed predicted contact windows; NEVER fabricate a full H300 exam."""

from typing import Any

import numpy as np

from rosclaw_soccer.rsi.contextual_first_touch_option import first_touch_reward


def predicted_touch_window(
    prediction: dict[str, Any], *, ball_address: int, root_address: int, start_control: int
) -> dict[str, Any]:
    n = prediction["controls"]
    if (
        type(n) is not int
        or not 1 <= n <= 64
        or type(start_control) is not int
        or not 0 <= start_control < 300
    ):
        raise ValueError("actual pending formal-task boundary and bounded prediction required")
    q, force = prediction["qpos"], prediction["ball_forces"]
    if (
        q.ndim != 2
        or len(q) != n * 10
        or force.shape != (n * 10, 1, 3)
        or q.dtype != np.float64
        or force.dtype != np.float64
        or not np.isfinite(q).all()
        or not np.isfinite(force).all()
        or (force < 0).any()
        or start_control + n > 300
        or root_address < 0
        or root_address + 7 > q.shape[1]
        or ball_address < 0
        or ball_address + 3 > q.shape[1]
    ):
        raise ValueError("finite unpadded native prediction inside formal deadline required")
    pelvis = q[:, root_address : root_address + 7]
    quat = pelvis[:, 3:7]
    if not np.allclose(np.linalg.norm(quat, axis=1), 1.0, rtol=0, atol=1e-5):
        raise ValueError("normalized actual predicted body orientation required")
    healthy = bool(
        (pelvis[:, 2] >= 0.65).all() and (1 - 2 * (quat[:, 1] ** 2 + quat[:, 2] ** 2) >= 0.9).all()
    )
    clean = bool(not (force[:, 0, 2] > 1).any())
    ball = q[9::10, ball_address : ball_address + 3]
    excursion = float(abs(ball[:, 1]).max())
    by_control = force[:, 0].reshape(n, 10, 3).max(axis=1)
    foot = np.flatnonzero((by_control[:, :2] > 1).any(axis=1))
    first = int(foot[0]) if len(foot) else None
    complete = first is not None and first + 60 < n
    forward = (
        float(ball[first + 60, 0] - ball[first, 0]) if first is not None and complete else None
    )
    lateral = (
        float(ball[first + 60, 1] - ball[first, 1]) if first is not None and complete else None
    )
    passes = bool(
        healthy
        and clean
        and excursion <= 4
        and forward is not None
        and forward >= 1.0
        and lateral is not None
        and abs(lateral) / max(forward, 0.01) <= 0.3
    )
    reward = -10.0 if not healthy or not clean else -3.0 if first is None else -1.0
    if healthy and clean and complete:
        assert first is not None
        reward = first_touch_reward(
            dict(
                first_contact_frame=start_control + first,
                contact_body_indices=np.flatnonzero(by_control[first] > 1).tolist(),
                minimum_pelvis_z_m=float(pelvis[:, 2].min()),
                max_lateral_excursion_m=excursion,
                forward_60_m=forward,
                lateral_60_m=lateral,
            )
        )
    return dict(
        window_pass=passes,
        quality_observed=complete,
        foot_observed=first is not None,
        first_foot_global_control=None if first is None else start_control + first,
        forward_60_m=forward,
        lateral_60_m=lateral,
        predicted_body_healthy=healthy,
        predicted_nonfoot_clean=clean,
        maximum_predicted_lateral_excursion_m=excursion,
        reward=float(reward),
        full_h300_task_pass_proven=False,
        requires_clean_actual_precontact_prefix=True,
    )
