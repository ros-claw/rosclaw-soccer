"""Current SIM state for future release learning, not motion or training approval.

Reuses112 measured physical body/ball/nominal-target features. Adds six past
contact-force magnitudes, the measured contact side and the current task clock.
Ball state is SIM ground truth, NOT real-robot proprioception/vision evidence.
No outcome, source case id, future contact time or future trajectory is input.
"""

from typing import Any

import numpy as np

from rosclaw_soccer.rsi.body_ball_response_features import measured_body_ball_features

WIDTH = 121


def current_postcontact_features(
    qpos: np.ndarray,
    qvel: np.ndarray,
    current_nominal_target_rad: np.ndarray,
    *,
    frame: int,
    previous_contact_forces: np.ndarray,
) -> np.ndarray:
    if type(frame) is not int or not 30 <= frame < 300:
        raise ValueError("actual before-action H300 postcontact frame required")
    for a, shape in (
        (qpos, (43,)),
        (qvel, (41,)),
        (current_nominal_target_rad, (29,)),
        (previous_contact_forces, (6,)),
    ):
        if (
            type(a) is not np.ndarray
            or a.shape != shape
            or a.dtype != np.float64
            or not np.isfinite(a).all()
            or np.max(abs(a)) > 1e6
        ):
            raise ValueError("whole bounded finite current float64 SIM observation required")
    if np.any(previous_contact_forces < 0) or not np.any(previous_contact_forces[:2] > 1):
        raise ValueError("measured completed foot-contact bin required, not predicted contact")
    base = measured_body_ball_features(qpos[None], qvel[None], current_nominal_target_rad[None])[0]
    side = np.zeros(2, dtype=np.float32)
    side[int(np.argmax(previous_contact_forces[:2]))] = 1
    result: np.ndarray = np.concatenate(
        (base, np.log1p(previous_contact_forces) / np.log(1001.0), side, [(300 - frame) / 300])
    ).astype(np.float32)
    if result.shape != (WIDTH,) or not np.isfinite(result).all():
        raise ValueError("complete finite current postcontact learning features required")
    result.flags.writeable = False
    return result


def observation_contract() -> dict[str, Any]:
    return dict(
        schema="soccer.rsi.current_postcontact_learning_observation.v1",
        width=WIDTH,
        measured_body_ball_features_prefix=112,
        measured_body_ball_feature_schema="measured_body_ball_features",
        canonical_qpos_width=43,
        canonical_qvel_width=41,
        nominal_target_width=29,
        appended_previous_contact_forces=6,
        force_transform="log1p(force_n)/log(1001)",
        appended_measured_contact_side_onehot=2,
        appended_remaining_time_fraction=True,
        horizon_frames=300,
        nominal_target_epoch="current before action",
        canonical_state_epoch="current integrated qpos/qvel, not cached-COM pose",
        ball_state="SIM ground truth, no real perception qualification",
        future_outcome_or_event_time_used=False,
        learning_or_execution_authorized=False,
        hardware_authorized=False,
    )
