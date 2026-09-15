from dataclasses import replace

import numpy as np
import pytest
from test_receiving_frame import build
from test_s368_recurrent_receiver import CONFIG, POLICY, observation

from rosclaw_soccer.providers.g1.recurrent_receiver import G1RecurrentReceiver
from rosclaw_soccer.sim.contracts import hash_bytes
from rosclaw_soccer.training.receiving_frame import (
    CANONICAL_POSE_RECEIVER_CONTRACT,
    CANONICAL_RECEIVER_CONTRACT,
    canonical_receiving_state,
    receiving_frame_translation,
)
from rosclaw_soccer.training.receiving_sampler import ReceivingSampler


def physical_pair():
    original = observation()
    q, v = np.array(original.qpos), np.array(original.qvel)
    q[:3] = [4.25, 1.1, 0.78]
    q[3:7] = [0.0, 0.0, 0.0, 1.0]
    q[36:39] = [4.92, -0.49, 0.115]
    v[:6] = [0.3, -0.2, 0.1, 0.4, 0.1, -0.2]
    v[35:38] = [0.6, 0.2, 0.0]
    p, w = canonical_receiving_state(q, v, half_turn=True, translation_xy_m=(6.0, 0.0))
    return original, q, v, p, w


def test_state_projection_maps_only_world_fields_and_preserves_source():
    _, q, v, p, w = physical_pair()
    assert p[0] == 1.75 and p[1] == -1.1
    np.testing.assert_array_equal(p[7:36], q[7:36])
    np.testing.assert_array_equal(w[3:35], v[3:35])
    np.testing.assert_array_equal(w[38:41], v[38:41])
    np.testing.assert_array_equal(w[[0, 1, 35, 36]], -v[[0, 1, 35, 36]])
    assert q[0] == 4.25 and v[0] == 0.3
    a, b = canonical_receiving_state(q, v, half_turn=False, translation_xy_m=(6.0, 0.0))
    np.testing.assert_array_equal(a, q)
    np.testing.assert_array_equal(b, v)
    assert not np.shares_memory(q, a) and not np.shares_memory(v, b)
    a, b = canonical_receiving_state(p, w, half_turn=True, translation_xy_m=(6.0, 0.0))
    # Two half-turns negate a quaternion: same orientation, not identical signs.
    expected = q.copy()
    expected[3:7] *= -1
    expected[39:43] *= -1
    np.testing.assert_allclose(a, expected, atol=1e-15)
    np.testing.assert_array_equal(b, v)


@pytest.mark.parametrize(
    "translation", [None, [6.0, 0.0], (True, 0.0), (6.0,), (np.nan, 0), (201, 0)]
)
def test_translation_must_be_explicit_bounded_immutable(translation):
    with pytest.raises(ValueError):
        receiving_frame_translation(translation)


@pytest.mark.parametrize(
    "fault", ["shape", "dtype", "nan", "bound", "body_quaternion", "ball_quaternion", "flag"]
)
def test_damaged_state_projection_rejected(fault):
    _, q, v, _, _ = physical_pair()
    flag = True
    if fault == "shape":
        q = q[:42]
    elif fault == "dtype":
        v = v.astype(np.float32)
    elif fault == "nan":
        v[0] = np.nan
    elif fault == "bound":
        q[0] = 10001
    elif fault == "body_quaternion":
        q[3:7] = 0
    elif fault == "ball_quaternion":
        q[39:43] = 0
    else:
        flag = 1
    with pytest.raises(ValueError):
        canonical_receiving_state(q, v, half_turn=flag, translation_xy_m=(6.0, 0.0))


def receiver(path, flag, translation=(6.0, 0.0), cls=G1RecurrentReceiver):
    return cls(
        path,
        **({"seed": 2242} if cls is ReceivingSampler else {}),
        agent_id="blue.playmaker",
        expected_actor_hash=hash_bytes(path.read_bytes()),
        foundation_hash=POLICY,
        foundation_config_hash=CONFIG,
        observation_contract=CANONICAL_POSE_RECEIVER_CONTRACT,
        canonical_half_turn=flag,
        canonical_translation_xy_m=translation,
    )


def test_prequantization_restores_exact_features_and_actions_in_numeric_counterexample(tmp_path):
    legacy = build(tmp_path, "recurrent_receiver_world_heading_135_float32.v2")
    old_rotated = build(tmp_path, CANONICAL_RECEIVER_CONTRACT, True)
    path = tmp_path / "heading.npz"
    unrotated = receiver(path, False)
    rotated = receiver(path, True)
    original, q, v, p, w = physical_pair()
    a = replace(original, qpos=tuple(map(float, q)), qvel=tuple(map(float, v)))
    b = replace(original, qpos=tuple(map(float, p)), qvel=tuple(map(float, w)))
    results = []
    for motor, obs in ((legacy, a), (unrotated, a), (rotated, b), (old_rotated, b)):
        motor.begin_skill(obs)
        results.append(motor.propose(obs))
    np.testing.assert_array_equal(legacy.last_observation, unrotated.last_observation)
    np.testing.assert_array_equal(legacy.last_observation, rotated.last_observation)
    assert results[0] == results[1] == results[2]
    assert not np.array_equal(legacy.last_observation, old_rotated.last_observation)
    assert len({m.contract_hash for m in (legacy, unrotated, rotated, old_rotated)}) == 4


def test_sampling_and_frame_translation_are_explicitly_bound(tmp_path):
    build(tmp_path, CANONICAL_RECEIVER_CONTRACT, True)
    path = tmp_path / "heading.npz"
    a = receiver(path, True)
    b = receiver(path, True, cls=ReceivingSampler)
    obs, _, _, q, v = physical_pair()
    obs = replace(obs, qpos=tuple(map(float, q)), qvel=tuple(map(float, v)))
    for motor in (a, b):
        motor.begin_skill(obs)
        motor.propose(obs)
    np.testing.assert_array_equal(a.last_observation, b.sampled_rollout()["obs"])
    assert a.contract_hash != receiver(path, True, translation=(5.0, 0.0)).contract_hash
    with pytest.raises(ValueError, match="translation"):
        receiver(path, True, translation=None)
    with pytest.raises(ValueError, match="boolean"):
        receiver(path, None)
    with pytest.raises(ValueError, match="pose contract"):
        G1RecurrentReceiver(
            path,
            agent_id="blue.playmaker",
            expected_actor_hash=hash_bytes(path.read_bytes()),
            foundation_hash=POLICY,
            foundation_config_hash=CONFIG,
            observation_contract=CANONICAL_RECEIVER_CONTRACT,
            canonical_half_turn=True,
            canonical_translation_xy_m=(6.0, 0.0),
        )
