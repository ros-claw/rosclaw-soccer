import json

import numpy as np
import pytest

from rosclaw_soccer.providers.g1.joint_contract import G1_DDS_JOINT_NAMES
from rosclaw_soccer.providers.g1.keeper_muscle_actor import (
    CONTRACT_HASH,
    KeeperMuscleActor,
    muscle_observation,
)


def artifact():
    return dict(
        schema="keeper-muscle-bc.v1",
        activation_ceiling="SIM_ONLY",
        promotion_authorized=False,
        observation_contract_hash=CONTRACT_HASH,
        joint_names=list(G1_DDS_JOINT_NAMES[15:]),
        layers=[
            dict(weight=np.zeros(shape).tolist(), bias=np.zeros(shape[0]).tolist())
            for shape in ((64, 65), (64, 64), (14, 64))
        ],
    )


def test_numeric_actor_arm_only_bounded_and_content_bound(tmp_path):
    path = tmp_path / "actor.json"
    payload = artifact()
    path.write_text(json.dumps(payload))
    actor = KeeperMuscleActor(path)
    np.testing.assert_array_equal(actor.target(np.zeros(65)), np.zeros(14))
    payload["layers"][-1]["bias"] = [20.0] * 14
    path.write_text(json.dumps(payload))
    changed = KeeperMuscleActor(path)
    assert changed.policy_hash != actor.policy_hash
    assert np.max(np.abs(changed.target(np.ones(65)))) <= 3
    for invalid in (np.zeros(64), np.full(65, np.nan), np.full(65, 5.1)):
        with pytest.raises(ValueError):
            actor.target(invalid)


@pytest.mark.parametrize(
    "key,value",
    [
        ("activation_ceiling", "REAL"),
        ("promotion_authorized", True),
        ("observation_contract_hash", "different"),
        ("joint_names", []),
        ("layers", []),
    ],
)
def test_rejects_contract_or_authority_change(tmp_path, key, value):
    payload = artifact()
    payload[key] = value
    path = tmp_path / "actor.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        KeeperMuscleActor(path)


def test_rejects_nonfinite_weights(tmp_path):
    payload = artifact()
    payload["layers"][0]["weight"][0][0] = float("nan")
    path = tmp_path / "actor.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        KeeperMuscleActor(path)


@pytest.mark.parametrize("payload", [[], {"schema": "invalid"}, None])
def test_rejects_malformed_artifact_objects(tmp_path, payload):
    path = tmp_path / "actor.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        KeeperMuscleActor(path)


def test_distillation_rejects_unbound_teacher_before_loading_arrays(tmp_path):
    pytest.importorskip("torch")
    from rosclaw_soccer.training.keeper_muscle_distillation import train

    source = tmp_path / "source"
    source.mkdir()
    cases = {}
    for lane in ("center-channel", "left-channel", "right-channel", "far-left-channel"):
        (source / f"{lane}-trajectory.npz").write_bytes(b"unbound fixture")
        cases[lane] = dict(
            passed=True, trajectory_hash="sha256:wrong", first=dict(gates=dict(physical_save=True))
        )
    (source / "evidence.json").write_text(
        json.dumps(dict(passed=True, activation_ceiling="SIM_ONLY", cases=cases))
    )
    with pytest.raises(ValueError, match="not bound"):
        train(source, tmp_path / "output", epochs=1)


def test_observation_order_scaling_and_finite_guard():
    observation = muscle_observation(
        np.array((0.25, -0.2, 1.5)), 0.8, np.array((0, 0, -1)), np.ones(29), np.ones(29) * 10
    )
    np.testing.assert_allclose(observation[:7], (0.5, -0.4, 1.5, 0.8, 0, 0, -1))
    np.testing.assert_allclose(observation[7:36], 1 / 3)
    np.testing.assert_allclose(observation[36:], 0.5)
    with pytest.raises(ValueError):
        muscle_observation(np.zeros(3), float("nan"), np.zeros(3), np.zeros(29), np.zeros(29))


def test_physics_reward_penalizes_falls_and_does_not_equal_save_gate():
    from rosclaw_soccer.training.keeper_muscle_evolution import physical_reward

    result = dict(
        physical_safe=True,
        minimum_pelvis_m=0.75,
        peak_tilt_rad=0.2,
        closest_incoming_glove_surface_m=0.1,
        outward_speed_mps=0,
        first_robot_contact_glove=False,
        completed_hand_save=False,
        stable_save=False,
        goal_crossed=True,
    )
    assert physical_reward(result) < 0
    result.update(closest_incoming_glove_surface_m=-0.01, first_robot_contact_glove=True)
    assert physical_reward(result) == 0
    assert result["completed_hand_save"] is False
    result.update(
        completed_hand_save=True, stable_save=True, outward_speed_mps=2, goal_crossed=False
    )
    assert physical_reward(result) > 5
    result["minimum_pelvis_m"] = 0.2
    assert physical_reward(result) == -10
    result["peak_tilt_rad"] = float("nan")
    with pytest.raises(ValueError):
        physical_reward(result)


def test_retention_null_direction_preserves_anchor_outputs(tmp_path):
    from rosclaw_soccer.training.keeper_muscle_evolution import retention_null_direction

    payload = artifact()
    rng = np.random.default_rng(230)
    for layer in payload["layers"]:
        layer["weight"] = (rng.normal(size=np.shape(layer["weight"])) * 0.15).tolist()
    path = tmp_path / "actor.json"
    path.write_text(json.dumps(payload))
    actor = KeeperMuscleActor(path)
    anchors, novel = rng.normal(size=(10, 65)), rng.normal(size=(8, 65))
    direction, report = retention_null_direction(actor, anchors, novel)
    assert report["anchor_projection_max"] < 1e-10
    weights = np.asarray(payload["layers"][-1]["weight"])
    bias = np.asarray(payload["layers"][-1]["bias"])
    delta = rng.normal(size=14)[:, None] * direction
    payload["layers"][-1]["weight"] = (weights + delta[:, :64]).tolist()
    payload["layers"][-1]["bias"] = (bias + delta[:, 64]).tolist()
    path.write_text(json.dumps(payload))
    changed = KeeperMuscleActor(path)
    for obs in anchors:
        np.testing.assert_allclose(changed.target(obs), actor.target(obs), atol=1e-10)
    assert max(np.max(abs(changed.target(v) - actor.target(v))) for v in novel) > 0.01
    with pytest.raises(ValueError):
        retention_null_direction(actor, np.full((10, 65), np.nan), novel)
