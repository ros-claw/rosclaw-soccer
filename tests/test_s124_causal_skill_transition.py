from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from rosclaw_soccer.growth.causal_skill_transition import (
    CausalTransitionSample,
    causal_transition_features,
    fit_causal_skill_transition_actor,
    load_causal_skill_transition_actor,
    save_causal_skill_transition_actor,
)
from rosclaw_soccer.sim.contracts import hash_json


def _samples() -> tuple[CausalTransitionSample, ...]:
    rows = []
    for index, distance in enumerate((4.10, 4.25, 4.40, 4.55, 4.70, 4.85, 5.00, 5.15)):
        features = causal_transition_features(
            receiver_pelvis_world_m=np.asarray((0.0, 0.0, 0.75)),
            predecessor_pelvis_world_m=np.asarray((distance, -0.1, 0.75)),
            receiver_ball_local_m=np.asarray((distance - 1.2, 0.0, 0.115)),
            receiver_reception_target_local_m=np.asarray((1.25, 0.0, 0.115)),
            receiver_shot_target_local_m=np.asarray((7.5, -2.9, 1.75)),
            predecessor_swing_speed_scale=0.80,
            ball_ground_friction=0.10,
            predecessor_yaw_rad=np.pi,
            receiver_kick_foot="right",
        )
        rows.append(
            CausalTransitionSample(
                sample_id=f"sample-{index}",
                features=tuple(float(value) for value in features),
                optimal_trigger_policy_frame=80 + 2 * index,
                source_trajectory_hash=str(hash_json({"sample": index})),
                safe=True,
            )
        )
    return tuple(rows)


def test_causal_transition_actor_is_bounded_and_json_only(tmp_path: Path) -> None:
    actor = fit_causal_skill_transition_actor(
        _samples(),
        source_stage_hash=str(hash_json({"source": "s123"})),
        seed=124,
        epochs=300,
    )
    path = tmp_path / "actor.json"
    save_causal_skill_transition_actor(actor, path)
    loaded = load_causal_skill_transition_actor(path)
    decision = loaded.decide(np.asarray(_samples()[3].features, dtype=np.float64))

    assert loaded.actor_hash == actor.actor_hash
    assert decision.accepted
    assert abs(decision.residual_frames) <= actor.maximum_trigger_residual_frames
    assert actor.minimum_trigger_policy_frame <= decision.trigger_policy_frame
    assert decision.trigger_policy_frame <= actor.maximum_trigger_policy_frame
    assert json.loads(path.read_text())["serialized_executable_code"] is False


def test_causal_transition_ood_falls_back_to_frozen_parent() -> None:
    actor = fit_causal_skill_transition_actor(
        _samples(),
        source_stage_hash=str(hash_json({"source": "s123"})),
        seed=125,
        epochs=100,
    )
    observation = np.asarray(_samples()[0].features, dtype=np.float64)
    observation[0] = 100.0

    decision = actor.decide(observation)

    assert not decision.accepted
    assert decision.residual_frames == 0
    assert decision.trigger_policy_frame == actor.parent_trigger_policy_frame


def test_causal_transition_never_extrapolates_beyond_training_trigger_hull() -> None:
    actor = fit_causal_skill_transition_actor(
        _samples(),
        source_stage_hash=str(hash_json({"source": "s123"})),
        seed=127,
        epochs=100,
    )
    observation = np.asarray(_samples()[3].features, dtype=np.float64)

    late = replace(actor, output_bias=100.0).decide(observation)
    early = replace(actor, output_bias=-100.0).decide(observation)

    assert late.trigger_policy_frame == actor.training_trigger_maximum
    assert early.trigger_policy_frame == actor.training_trigger_minimum


def test_causal_transition_artifact_rejects_tampering(tmp_path: Path) -> None:
    actor = fit_causal_skill_transition_actor(
        _samples(),
        source_stage_hash=str(hash_json({"source": "s123"})),
        seed=126,
        epochs=100,
    )
    path = tmp_path / "actor.json"
    save_causal_skill_transition_actor(actor, path)
    payload = json.loads(path.read_text())
    payload["output_bias"] += 0.5
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="hash does not match"):
        load_causal_skill_transition_actor(path)


def test_causal_transition_rejects_unsafe_training_sample() -> None:
    samples = list(_samples())
    samples[0] = replace(samples[0], safe=False)

    with pytest.raises(ValueError, match="safe samples"):
        fit_causal_skill_transition_actor(
            tuple(samples),
            source_stage_hash=str(hash_json({"source": "s123"})),
            epochs=50,
        )


def test_causal_transition_actor_locks_schema_contract() -> None:
    actor = fit_causal_skill_transition_actor(
        _samples(),
        source_stage_hash=str(hash_json({"source": "s123"})),
        epochs=50,
    )

    with pytest.raises(ValueError, match="bounded SIM-only contract"):
        replace(actor, schema_version="rosclaw.growth.g1_causal_skill_transition_actor.v999")
