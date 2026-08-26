from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from rosclaw_soccer.skills.goalkeeper_v2.observations import (
    GoalkeeperActorObservation,
    GoalkeeperObservationSpec,
)
from rosclaw_soccer.skills.goalkeeper_v2.policy import (
    GoalkeeperActorArtifact,
    GoalkeeperDenseLayer,
    NumpyGoalkeeperActor,
    load_goalkeeper_actor_artifact,
    save_goalkeeper_actor_artifact,
)

_HASH = "sha256:" + "2" * 64


def _artifact() -> GoalkeeperActorArtifact:
    spec = GoalkeeperObservationSpec()
    first = np.zeros((len(spec.actor_names), 4), dtype=np.float64)
    first[0, 0] = 1.0
    second = np.zeros((4, 30), dtype=np.float64)
    second[0, 0] = 1.0
    return GoalkeeperActorArtifact(
        policy_id="goalkeeper.g1",
        generation=1,
        parent_policy_hash=_HASH,
        body_hash=_HASH,
        actor_observation_contract_hash=spec.actor_contract_hash,
        motion_library_hash=_HASH,
        training_run_hash=_HASH,
        layers=(
            GoalkeeperDenseLayer(
                weights=tuple(tuple(float(value) for value in row) for row in first),
                bias=(0.0,) * 4,
                activation="tanh",
            ),
            GoalkeeperDenseLayer(
                weights=tuple(tuple(float(value) for value in row) for row in second),
                bias=(0.0,) * 30,
                activation="tanh",
            ),
        ),
        maximum_lateral_speed_mps=0.6,
        maximum_joint_residual_rad=(0.1,) * 29,
    )


def _observation(value: float = 1.0) -> GoalkeeperActorObservation:
    spec = GoalkeeperObservationSpec()
    values = np.zeros(len(spec.actor_names), dtype=np.float64)
    values[0] = value
    return GoalkeeperActorObservation(
        values=tuple(float(item) for item in values),
        actor_contract_hash=spec.actor_contract_hash,
        ball_history_ready=True,
        estimated_ball_velocity_mps=(-1.0, 0.0, 0.0),
        estimated_intercept=(1.0, 0.0, 0.3),
        intercept_confidence=1.0,
        estimated_target_region=(0.0, 0.0, 0.0, 0.0, 1.0, 0.0),
        observed_flight_start_sec=0.0,
    )


def test_numpy_actor_is_bounded_causal_and_roundtrips(tmp_path: Path) -> None:
    artifact = _artifact()
    action = NumpyGoalkeeperActor(artifact).action(_observation())

    assert 0.0 < action.lateral_velocity_mps < artifact.maximum_lateral_speed_mps
    assert 0.0 < action.operational_space_reach_fraction <= 1.0
    assert max(abs(item) for item in action.joint_position_residual_rad) <= 0.1
    assert not artifact.deployed_actor_uses_privileged_critic
    assert not artifact.operational_space_reach_enabled
    assert artifact.operational_space_reach_ramp_sec == 0.18
    assert artifact.operational_space_memory_decay == 0.75
    assert artifact.operational_space_memory_maximum_rad == 0.22

    path = tmp_path / "evidence" / "actor.json"
    save_goalkeeper_actor_artifact(
        artifact,
        path,
        source_checkout=tmp_path / "checkout",
    )
    assert load_goalkeeper_actor_artifact(path) == artifact


def test_actor_rejects_contract_mismatch_and_tampering(tmp_path: Path) -> None:
    artifact = _artifact()
    bad = _observation()
    object.__setattr__(bad, "actor_contract_hash", "sha256:" + "3" * 64)
    with pytest.raises(ValueError, match="contract hash mismatch"):
        NumpyGoalkeeperActor(artifact).action(bad)

    path = tmp_path / "actor.json"
    save_goalkeeper_actor_artifact(
        artifact,
        path,
        source_checkout=tmp_path / "checkout",
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["layers"][0]["weights"][0][0] = 99.0
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="content hash mismatch"):
        load_goalkeeper_actor_artifact(path)


def test_arm_intent_can_activate_reach_without_lateral_motion() -> None:
    artifact = _artifact()
    first, second = artifact.layers
    second_weights = np.asarray(second.weights, dtype=np.float64)
    second_weights[:, 0] = 0.0
    second_weights[0, 1 + 15] = 1.0
    arm_only = GoalkeeperActorArtifact(
        **{
            **artifact.__dict__,
            "layers": (
                first,
                GoalkeeperDenseLayer(
                    weights=tuple(tuple(float(value) for value in row) for row in second_weights),
                    bias=second.bias,
                    activation=second.activation,
                ),
            ),
        }
    )

    action = NumpyGoalkeeperActor(arm_only).action(_observation())

    assert action.lateral_velocity_mps == 0.0
    assert action.operational_space_reach_fraction > 0.0


def test_goalkeeper_actor_artifact_allows_cerebellum_owned_zero_leg_residuals() -> None:
    artifact = GoalkeeperActorArtifact(
        **{
            **_artifact().__dict__,
            "maximum_joint_residual_rad": (0.0,) * 12 + (0.1,) * 17,
        }
    )

    action = NumpyGoalkeeperActor(artifact).action(_observation())

    assert action.joint_position_residual_rad[:12] == (0.0,) * 12


def test_goalkeeper_actor_artifact_rejects_impulsive_reach_ramp() -> None:
    with pytest.raises(ValueError, match="outside bounds"):
        GoalkeeperActorArtifact(
            **{
                **_artifact().__dict__,
                "operational_space_reach_ramp_sec": 0.02,
            }
        )
