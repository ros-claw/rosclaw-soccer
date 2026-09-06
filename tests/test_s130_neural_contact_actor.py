from __future__ import annotations

import inspect

import numpy as np

from rosclaw_soccer.growth.neural_contact_actor import (
    NEURAL_CONTACT_FEATURE_NAMES,
    G1NeuralContactActor,
    evaluate_neural_contact_actor,
    neural_contact_features,
)
from rosclaw_soccer.skills.team.shared_world import simulate_shared_world


def _actor() -> G1NeuralContactActor:
    feature_count = len(NEURAL_CONTACT_FEATURE_NAMES)
    hidden = 16
    minimum = [-10.0] * feature_count
    maximum = [10.0] * feature_count
    minimum[1:4] = [7.0, -3.0, -1.0]
    maximum[1:4] = [9.0, 3.0, 3.0]
    return G1NeuralContactActor(
        body_hash="sha256:" + "1" * 64,
        implementation_hash="sha256:" + "2" * 64,
        source_evidence_hashes=("sha256:" + "3" * 64,),
        training_snapshot_hash="sha256:" + "4" * 64,
        feature_center=(0.0,) * feature_count,
        feature_scale=(1.0,) * feature_count,
        feature_minimum=tuple(minimum),
        feature_maximum=tuple(maximum),
        hidden_one_weights=tuple((0.0,) * feature_count for _ in range(hidden)),
        hidden_one_bias=(0.0,) * hidden,
        hidden_two_weights=tuple((0.0,) * hidden for _ in range(hidden)),
        hidden_two_bias=(0.0,) * hidden,
        output_weights=tuple((0.0,) * hidden for _ in range(29)),
        output_bias=(1.0,) * 29,
        minimum_torque_nm=(-5.0,) * 29,
        maximum_torque_nm=(5.0,) * 29,
        minimum_target_velocity_xyz_mps=(7.0, -3.0, -1.0),
        maximum_target_velocity_xyz_mps=(9.0, 3.0, 3.0),
        minimum_phase_offset_frames=-5,
        maximum_phase_offset_frames=8,
        maximum_normalized_ood_distance=0.75,
        training_sample_count=64,
        training_trajectory_count=2,
        failed_trajectory_count=1,
        training_rmse_nm=1.0,
    )


def test_neural_contact_actor_emits_bounded_direct_torque_and_fails_closed() -> None:
    actor = _actor()
    features = neural_contact_features(
        phase_offset_frames=0.0,
        target_velocity_xyz_mps=(8.0, 0.0, 1.0),
        ball_local_position_m=(0.0, 0.0, 0.1),
        ball_local_velocity_mps=(0.0, 0.0, 0.0),
        joint_position_rad=np.zeros(29),
        joint_velocity_rad_s=np.zeros(29),
    )
    effect = evaluate_neural_contact_actor(actor=actor, features=features)
    assert effect.active and effect.supported
    np.testing.assert_allclose(effect.torque, 1.0)

    outside = features.copy()
    outside[0] = 20.0
    rejected = evaluate_neural_contact_actor(actor=actor, features=outside)
    assert not rejected.active and not rejected.supported
    np.testing.assert_array_equal(rejected.torque, 0.0)


def test_shared_world_exposes_atomic_neural_contact_commitment() -> None:
    parameters = inspect.signature(simulate_shared_world).parameters
    assert "simulation_kwargs" in parameters
    source = inspect.getsource(simulate_shared_world.__globals__["_simulate_shared_world"])
    for name in (
        "shooter_neural_contact_actor_path",
        "shooter_neural_contact_policy_frame",
        "shooter_neural_contact_target_velocity_xyz_mps",
    ):
        assert name in source
