import json

import numpy as np
import pytest

from rosclaw_soccer.providers.g1.proprioceptive_router import (
    G1ProprioceptiveApproachRouter,
    g1_approach_proprioception,
)
from rosclaw_soccer.sim.contracts import hash_bytes

BODY = "sha256:" + "a" * 64
REFERENCES = "sha256:" + "b" * 64


def state():
    q = np.zeros(43)
    q[2], q[3], q[36], q[38], q[39] = 0.75, 1, 0.6, 0.115, 1
    return q, np.zeros(41)


def artifact(tmp_path):
    shapes = {
        "0.weight": (128, 74),
        "0.bias": (128,),
        "2.weight": (64, 128),
        "2.bias": (64,),
        "4.weight": (2, 64),
        "4.bias": (2,),
    }
    weights = {k: np.zeros(v, dtype=np.float32) for k, v in shapes.items()}
    weights["4.bias"][1] = 0.75
    path = tmp_path / "router.json"
    np.savez_compressed(path.with_suffix(".npz"), **weights)
    metadata = dict(
        schema="rosclaw_soccer.g1_proprioceptive_router.v1",
        activation_ceiling="SIM_ONLY",
        body_actor_hash=BODY,
        reference_library_hash=REFERENCES,
        feature_contract="g1_course_proprioception_74.v1",
        reference_indices=[14, 40],
        feature_mean=[0.0] * 74,
        feature_scale=[1.0] * 74,
        weights_hash=hash_bytes(path.with_suffix(".npz").read_bytes()),
    )
    path.write_text(json.dumps(metadata))
    return path, metadata


def load(path):
    return G1ProprioceptiveApproachRouter(
        path, expected_body_actor_hash=BODY, expected_reference_library_hash=REFERENCES
    )


def test_proprioception_observes_limbs_and_motion_without_mutating_world():
    q, v = state()
    q[7], v[6], v[0], v[3], v[35] = 0.3, 2, 0.7, 0.5, 0.2
    before_q, before_v = q.copy(), v.copy()
    features = g1_approach_proprioception(q, v)
    assert features.shape == (74,)
    assert features.dtype == np.float32
    np.testing.assert_allclose(
        features[[0, 29, 61, 64, 67, 70, 73]], [0.3, 0.2, 0.7, 0.1, 0.6, 0.2, 0.75]
    )
    np.testing.assert_array_equal(q, before_q)
    np.testing.assert_array_equal(v, before_v)


def test_proposal_is_content_bound_not_motion_or_success(tmp_path):
    path, _ = artifact(tmp_path)
    router = load(path)
    q, v = state()
    proposal = router.propose(course_qpos=q, course_qvel=v)
    assert proposal.reference_index == 40
    assert proposal.predicted_utility == 0.75
    assert proposal.activation_ceiling == "SIM_ONLY"
    assert proposal.policy_hash == hash_bytes(path.with_suffix(".npz").read_bytes())
    v[6] = 0.2
    assert (
        router.propose(course_qpos=q, course_qvel=v).observation_hash != proposal.observation_hash
    )


@pytest.mark.parametrize(
    "key,value",
    [
        ("body_actor_hash", REFERENCES),
        ("activation_ceiling", "REAL"),
        ("feature_contract", "arbitrary_pose"),
        ("feature_mean", [0.0] * 73),
        ("feature_scale", [0.0] * 74),
        ("feature_scale", [True] * 74),
        ("reference_indices", [14, 14]),
        ("weights_hash", None),
    ],
)
def test_manifest_rejects_ambiguity(tmp_path, key, value):
    path, metadata = artifact(tmp_path)
    metadata[key] = value
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError):
        load(path)


def test_rejects_invalid_or_extrapolated_body_observations(tmp_path):
    path, _ = artifact(tmp_path)
    router = load(path)
    q, v = state()
    for bad in (q[:-1], np.full(43, np.nan), np.ones(43, dtype=bool)):
        with pytest.raises(ValueError):
            router.propose(course_qpos=bad, course_qvel=v)
    q[3] = 2.0
    with pytest.raises(ValueError, match="quaternion"):
        router.propose(course_qpos=q, course_qvel=v)
    q[3] = 1.0
    v[0] = 9.0
    with pytest.raises(ValueError, match="domain"):
        router.propose(course_qpos=q, course_qvel=v)


def test_checkpoint_change_is_not_silently_loaded(tmp_path):
    path, _ = artifact(tmp_path)
    path.with_suffix(".npz").write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash"):
        load(path)


def test_integer_coordinates_do_not_wrap_before_feature_domain_check():
    q = np.zeros(43, dtype=np.int64)
    q[3], q[39] = 1, 1
    q[0], q[36] = -(2**62), 2**62
    features = g1_approach_proprioception(q, np.zeros(41))
    assert features[67] == pytest.approx(float(2**63))
