"""Synthetic velocity-native schema fixtures; no G1 physics/policy success."""

from types import SimpleNamespace

import numpy as np
import pytest
from test_g1_velocity_policy import parameters

from rosclaw_soccer.providers.g1.velocity_policy import observation98, phase_delta
from rosclaw_soccer.rsi.locomotion_response_protocol import response_courses
from rosclaw_soccer.rsi.velocity_policy_native_evidence import review_velocity_recording


def fixture():
    p = parameters()
    c = response_courses()[0]
    q, v = np.zeros((300, 43)), np.zeros((300, 41))
    q[:, 2] = 0.793
    q[:, 3] = q[:, 39] = 1.0
    q[:, 36:39] = [3.0, 3.0, 0.13]
    phase = phase_delta()
    features = []
    phases = []
    for f in range(300):
        obs, phase = observation98(
            q[f],
            v[f],
            np.zeros(3),
            np.zeros(3, dtype=np.float32),
            np.zeros(29, dtype=np.float32),
            phase,
            p,
        )
        features.append(obs)
        phases.append(phase)
    raw = dict(
        canonical_qpos=q,
        canonical_qvel=v,
        joint_target_rad=np.zeros((300, 29)),
        pd_requested_torque_nm=np.zeros((300, 10, 29)),
        torque_nm=np.zeros((300, 10, 29)),
        actual_actuator_force_nm=np.zeros((300, 10, 29)),
        pelvis_z_per_substep_m=np.full((300, 10), 0.793),
        force_n=np.zeros((300, 6)),
        ball_position_after_step_m=q[:, 36:39].copy(),
        target_body_canonical_qpos_after_substep=np.repeat(q[:, None], 10, axis=1),
        target_body_canonical_qvel_after_substep=np.repeat(v[:, None], 10, axis=1),
        ball_contact_force_by_body_n=np.zeros((300, 10, 46)),
        velocity_policy_observation=np.asarray(features),
        velocity_policy_action_hg=np.zeros((300, 29), dtype=np.float32),
        command_body_mps=np.zeros((300, 3), dtype=np.float32),
        command_world_mps=np.zeros((300, 3)),
        angular_velocity_body_radps=np.zeros((300, 3)),
        gait_phase=np.asarray(phases, dtype=np.float32),
    )
    sensor = SimpleNamespace(
        model=SimpleNamespace(nbody=46), angular_velocity_body=lambda q, v: np.zeros(3)
    )
    return raw, p, sensor, c


def test_full_current_features_target_native_pd_and_state_continuity_reviewed():
    raw, p, s, c = fixture()
    r = review_velocity_recording(raw, p, s, c)
    assert r["all_policy_features_and_native_pd_reconstructed"]
    assert not r["football_skill_proven"]
    assert r["new_native_steps"] == r["new_neural_calls"] == 0


@pytest.mark.parametrize(
    "kind", ["alias", "dtype", "history", "phase", "command", "pd", "continuity"]
)
def test_wrong_actor_codec_or_corrupt_native_control_fails_closed(kind):
    raw, p, s, c = fixture()
    if kind == "alias":
        raw["neural_action_isaac"] = raw.pop("velocity_policy_action_hg")
    elif kind == "dtype":
        raw["command_body_mps"] = raw["command_body_mps"].astype(np.float64)
    elif kind == "history":
        raw["velocity_policy_observation"][10, 69] = 1.0
    elif kind == "phase":
        raw["gait_phase"][3] = 0.4
    elif kind == "command":
        raw["command_world_mps"][5, 0] = 0.3
    elif kind == "pd":
        raw["pd_requested_torque_nm"][5, 3, 0] = 1.0
    else:
        raw["target_body_canonical_qpos_after_substep"][5, 9, 0] = 0.1
    with pytest.raises(ValueError):
        review_velocity_recording(raw, p, s, c)


def test_repeat_comparator_does_not_confuse_npz_container_hash_with_numerical_equality():
    from scripts.rsi_collect_velocity_policy_response import compare_velocity_result

    a = dict(native_hash="container1", native_steps=3000, response=[0.2, 0.0])
    b = {**a, "native_hash": "container2"}
    compare_velocity_result(a, b)
    with pytest.raises(ValueError):
        compare_velocity_result(a, {**b, "native_steps": 2990})
