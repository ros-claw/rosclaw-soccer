from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from rosclaw_soccer.growth.first_touch_interception import FirstTouchInterceptionConfig
from rosclaw_soccer.growth.reactive_route_actor import (
    REACTIVE_ROUTE_FEATURE_NAMES,
    ReactiveRouteSample,
    fit_reactive_route_actor,
)
from rosclaw_soccer.growth.temporal_route_actor import (
    G1TemporalRouteActor,
    TemporalRouteSequence,
    fit_temporal_route_actor,
    load_route_actor,
    load_temporal_route_actor,
    save_temporal_route_actor,
)
from rosclaw_soccer.sim.contracts import hash_json
from rosclaw_soccer.skills.team.shared_world import G1ReactiveMovementConfig
from rosclaw_soccer.training.temporal_route_growth import (
    default_temporal_route_retention_manifest_v5,
    default_temporal_route_retention_manifest_v6,
    default_temporal_route_retention_manifest_v7,
    default_temporal_route_retention_manifest_v8,
)


def _sequences() -> tuple[TemporalRouteSequence, ...]:
    rows = []
    for sequence_index in range(8):
        features = []
        commands = []
        previous = np.zeros(2, dtype=np.float64)
        for frame in range(128):
            phase = 0.08 * frame + 0.2 * sequence_index
            observation = np.zeros(len(REACTIVE_ROUTE_FEATURE_NAMES), dtype=np.float64)
            observation[:4] = (
                np.cos(phase),
                np.sin(phase),
                0.1 * np.sin(phase),
                -0.1 * np.cos(phase),
            )
            observation[4:10] = 0.2 * observation[:6]
            observation[10:14] = (
                sequence_index % 2,
                (sequence_index + 1) % 2,
                sequence_index % 2,
                (sequence_index + 1) % 2,
            )
            command = 0.75 * previous + 0.25 * observation[:2] * 0.35
            features.append(tuple(float(value) for value in observation))
            commands.append((float(command[0]), float(command[1])))
            previous = command
        rows.append(
            TemporalRouteSequence(
                sequence_id=f"sequence-{sequence_index}",
                features=tuple(features),
                teacher_world_commands_xy_mps=tuple(commands),
            )
        )
    return tuple(rows)


def _actor() -> G1TemporalRouteActor:
    sequences = _sequences()
    source_stage_hash = hash_json({"stage": "s122"})
    samples = tuple(
        ReactiveRouteSample(
            episode_id=sequence.sequence_id,
            features=features,
            teacher_world_command_xy_mps=command,
        )
        for sequence in sequences
        for features, command in zip(
            sequence.features, sequence.teacher_world_commands_xy_mps, strict=True
        )
    )
    parent = fit_reactive_route_actor(samples, source_stage_hash=source_stage_hash)
    return fit_temporal_route_actor(
        sequences,
        source_stage_hash=source_stage_hash,
        source_actor_hash=parent.actor_hash,
        parent_actor=parent,
        hidden_size=12,
        epochs=120,
        learning_rate=8.0e-3,
        seed=123,
    )


def test_temporal_actor_round_trip_memory_and_support_gate(tmp_path: Path) -> None:
    actor = _actor()
    path = tmp_path / "temporal.json"
    save_temporal_route_actor(actor, path)
    loaded = load_temporal_route_actor(path)
    assert loaded == actor
    assert load_route_actor(path) == actor

    observation = np.asarray(_sequences()[0].features[10], dtype=np.float64)
    first = loaded.decide(observation)
    next_observation = np.asarray(_sequences()[0].features[11], dtype=np.float64)
    second = loaded.decide(next_observation, first.next_memory)
    stateless_second = loaded.decide(next_observation)
    assert first.accepted and second.accepted
    assert second.next_memory.feature_ema != pytest.approx(stateless_second.next_memory.feature_ema)
    assert np.linalg.norm(np.asarray(loaded.hidden_weights)[:, 14:]) > 0.0
    assert second.next_memory.previous_command_xy_mps == second.world_command_xy_mps

    rejected = loaded.decide(np.full(len(REACTIVE_ROUTE_FEATURE_NAMES), 100.0))
    assert not rejected.accepted
    assert rejected.world_command_xy_mps == (0.0, 0.0)


def test_temporal_actor_rejects_tampering_and_hardware_authority(tmp_path: Path) -> None:
    actor = _actor()
    path = tmp_path / "temporal.json"
    save_temporal_route_actor(actor, path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["hidden_weights"][0][0] += 0.25
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="hash does not match"):
        load_temporal_route_actor(path)

    values = actor.to_dict(include_hash=False)
    for key in (
        "input_names",
        "algorithm",
        "serialized_executable_code",
        "pixels_used_for_training",
        "current_stage_retention_evidence_used_for_training",
        "released_source_stage_evidence_used_for_training",
        "output_authority",
    ):
        values.pop(key)
    values["hardware_authorized"] = True
    with pytest.raises(ValueError, match="SIM-only contract"):
        G1TemporalRouteActor(**values)


def test_temporal_sequence_rejects_unordered_short_data() -> None:
    with pytest.raises(ValueError, match="malformed"):
        TemporalRouteSequence(
            sequence_id="short",
            features=((0.0,) * len(REACTIVE_ROUTE_FEATURE_NAMES),),
            teacher_world_commands_xy_mps=((0.0, 0.0),),
        )


def test_predictive_reception_and_continuous_braking_envelopes(tmp_path: Path) -> None:
    actor = _actor()
    path = tmp_path / "temporal.json"
    save_temporal_route_actor(actor, path)
    config = G1ReactiveMovementConfig(
        actor_artifact_path=str(path),
        actor_hash=actor.actor_hash,
        target_position_m=(5.0, -0.6, 0.0),
        action="pass",
        role="teammate",
    )
    assert config.maximum_velocity_braking_correction_mps == pytest.approx(0.18)
    assert config.diagonal_braking_target_dx_start_m < config.diagonal_braking_target_dx_full_m
    assert config.diagonal_braking_target_dy_start_m < config.diagonal_braking_target_dy_full_m
    assert FirstTouchInterceptionConfig(maximum_foot_ball_distance_m=1.20)
    with pytest.raises(ValueError, match="proximity"):
        FirstTouchInterceptionConfig(maximum_foot_ball_distance_m=1.51)
    with pytest.raises(ValueError, match="locomotion envelope"):
        G1ReactiveMovementConfig(
            actor_artifact_path=str(path),
            actor_hash=actor.actor_hash,
            target_position_m=(5.0, -0.6, 0.0),
            action="pass",
            role="teammate",
            diagonal_braking_target_dx_start_m=0.60,
            diagonal_braking_target_dx_full_m=0.60,
        )


def test_sealed_retention_generations_are_disjoint_and_sim_only() -> None:
    manifests = tuple(
        builder()
        for builder in (
            default_temporal_route_retention_manifest_v5,
            default_temporal_route_retention_manifest_v6,
            default_temporal_route_retention_manifest_v7,
            default_temporal_route_retention_manifest_v8,
        )
    )
    hashes = [case.case_hash for manifest in manifests for case in manifest.cases]
    ids = [case.scenario.scenario_id for manifest in manifests for case in manifest.cases]
    assert len(hashes) == len(set(hashes)) == 32
    assert len(ids) == len(set(ids)) == 32
    assert all(not manifest.training_access_allowed for manifest in manifests)
    assert all(manifest.activation_ceiling == "SIM_ONLY" for manifest in manifests)
