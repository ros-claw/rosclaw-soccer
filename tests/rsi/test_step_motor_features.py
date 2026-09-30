import copy

import numpy as np
import pytest

from rosclaw_soccer.rsi.step_motor_features import FEATURE_NAMES, feature_vector


def observation():
    return dict(
        joint_position=np.zeros(29),
        joint_velocity=np.zeros(29),
        root_pose_xyzw=np.asarray((0.0, 0.0, 0.8, 0.0, 0.0, 0.0, 1.0)),
        root_velocity_world=np.zeros(6),
        ball_position_world=np.asarray((1.0, 0.1, 0.11)),
        ball_velocity_world=np.asarray((-0.5, 0.0, 0.0)),
        geometry_position_world=np.zeros((4, 3)),
        nominal_target=np.zeros(29),
        previous_motor_delta=np.zeros(12),
        previous_contact_forces=np.zeros(6),
        frame=30,
    )


def test_complete_causal_feature_contract_has_no_episode_identity():
    assert len(FEATURE_NAMES) == 134
    values = feature_vector(**observation())
    assert values.shape == (134,)
    assert not any("seed" in n or "reward" in n or "future" in n for n in FEATURE_NAMES)


def test_field_translation_does_not_change_motor_features():
    obs = observation()
    translated = copy.deepcopy(obs)
    delta = np.asarray((10.0, -4.0, 0.0))
    translated["root_pose_xyzw"][:3] += delta
    translated["ball_position_world"] += delta
    translated["geometry_position_world"] += delta
    assert np.allclose(feature_vector(**obs), feature_vector(**translated), atol=1e-14, rtol=0)


def test_heading_rotation_does_not_change_motor_features():
    obs = observation()
    rotated = copy.deepcopy(obs)
    angle = 0.7
    r = np.asarray(
        ((np.cos(angle), -np.sin(angle), 0.0), (np.sin(angle), np.cos(angle), 0.0), (0.0, 0.0, 1.0))
    )
    rotated["root_pose_xyzw"][3:] = (0.0, 0.0, np.sin(angle / 2), np.cos(angle / 2))
    for key in ("ball_position_world", "ball_velocity_world"):
        rotated[key] = r @ rotated[key]
    rotated["geometry_position_world"] = rotated["geometry_position_world"] @ r.T
    assert np.allclose(feature_vector(**obs), feature_vector(**rotated), atol=1e-14, rtol=0)


@pytest.mark.parametrize(
    "field", ["joint_position", "root_velocity_world", "previous_contact_forces"]
)
def test_nonfinite_proprioception_fails_closed(field):
    obs = observation()
    obs[field][0] = float("nan")
    with pytest.raises(ValueError):
        feature_vector(**obs)


def test_bad_quaternion_and_future_frame_rejected():
    obs = observation()
    obs["root_pose_xyzw"][-1] = 2
    with pytest.raises(ValueError):
        feature_vector(**obs)
    obs = observation()
    obs["frame"] = 3000
    with pytest.raises(ValueError):
        feature_vector(**obs)
