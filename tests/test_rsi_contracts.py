from dataclasses import replace

import pytest

from rosclaw_soccer.rsi.contracts import (
    AthleteObservation,
    AthleticIntent,
    EpisodePartition,
    MotorAction,
    PhysicalEpisode,
    PolicyArtifact,
    validate_athlete_proposal,
)

H = "sha256:" + "a" * 64
J = "sha256:" + "b" * 64


def artifact() -> PolicyArtifact:
    return PolicyArtifact("g1.athlete", "sonic", H, H, H, H, H, H, H, H, "JOINT_TARGET", 29)


def observation() -> AthleteObservation:
    return AthleteObservation(
        "g1",
        H,
        H,
        50,
        (0.0, 0.0, 0.8),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0, 0.0),
        (0.0,) * 29,
        (0.0,) * 29,
        (True, False),
    )


def episode(partition: EpisodePartition = EpisodePartition.CONSUMED_DEV) -> PhysicalEpisode:
    return PhysicalEpisode(
        "r1.scene.001",
        partition,
        H,
        H,
        H,
        H,
        H,
        H,
        H,
        H,
        safe=True,
        task_success=True,
        teacher_active=False,
    )


def test_matching_simulation_proposal_is_only_a_typed_proposal():
    parent = artifact()
    obs = observation()
    action = MotorAction(H, parent.contract_hash, obs.frame, joint_target=(0.0,) * 29)
    validate_athlete_proposal(parent, obs, action)
    assert action.activation_ceiling == "SIM_ONLY"
    assert episode().contract_hash != replace(episode(), task_success=False).contract_hash
    latent_artifact = replace(parent, action_kind="MOTOR_LATENT", action_size=4)
    validate_athlete_proposal(
        latent_artifact,
        obs,
        MotorAction(H, latent_artifact.contract_hash, obs.frame, motor_latent=(0.0,) * 4),
    )
    with pytest.raises(ValueError):
        validate_athlete_proposal(
            latent_artifact,
            obs,
            MotorAction(H, latent_artifact.contract_hash, obs.frame, motor_latent=(0.0,) * 3),
        )


@pytest.mark.parametrize(
    "fault", ["foreign_body", "foreign_joint_map", "wrong_policy", "stale_frame", "wrong_joints"]
)
def test_foreign_or_stale_policy_cannot_consume_body_state(fault):
    parent = artifact()
    action = MotorAction(H, parent.contract_hash, 50, joint_target=(0.0,) * 29)
    if fault == "foreign_body":
        action = replace(action, body_hash=J)
    elif fault == "foreign_joint_map":
        with pytest.raises(ValueError):
            validate_athlete_proposal(parent, replace(observation(), joint_map_hash=J), action)
        return
    elif fault == "wrong_policy":
        action = replace(action, policy_hash=J)
    elif fault == "stale_frame":
        action = replace(action, frame=49)
    else:
        action = replace(action, joint_target=(0.0,) * 28)
    with pytest.raises(ValueError):
        validate_athlete_proposal(parent, observation(), action)


def test_exam_rejects_teacher_and_unsafe_success():
    with pytest.raises(ValueError):
        replace(episode(EpisodePartition.FRESH_HOLDOUT), teacher_active=True)
    with pytest.raises(ValueError):
        replace(episode(EpisodePartition.SEALED), safe=False)
    assert replace(episode(), teacher_active=True).partition is EpisodePartition.CONSUMED_DEV


def test_model_and_episode_refuse_unbound_or_nonfinite_evidence():
    with pytest.raises(ValueError):
        replace(artifact(), weights_hash="missing")
    with pytest.raises(ValueError):
        replace(observation(), joint_velocity=(float("nan"),) * 29)
    with pytest.raises(ValueError):
        replace(observation(), root_angular_velocity_rad_s=(0.0, float("inf"), 0.0))
    with pytest.raises(ValueError):
        replace(episode(), trajectory_hash="unverified")
    with pytest.raises(ValueError):
        MotorAction(H, H, 50, joint_target=(0.0,), motor_latent=(0.0,))
    with pytest.raises(ValueError):
        MotorAction(H, H, 50, joint_target=(0.0,), activation_ceiling="REAL")


def test_intent_rejects_nonfinite_or_boolean_motion():
    intent = AthleticIntent((1.0, 0.0), 0.0, 0.0, 0.75, "kick", "soccer")
    with pytest.raises(ValueError):
        replace(intent, body_height_m=float("nan"))
    with pytest.raises(ValueError):
        replace(intent, heading_rad=True)
