from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from rosclaw_soccer.growth.reactive_route_actor import (
    REACTIVE_ROUTE_FEATURE_NAMES,
    ReactiveRouteSample,
    fit_reactive_route_actor,
    load_reactive_route_actor,
    reactive_route_features,
    save_reactive_route_actor,
)
from rosclaw_soccer.growth.tactical_2v1 import TacticalAction
from rosclaw_soccer.sim.contracts import hash_json
from rosclaw_soccer.training.full_body_tactical_2v1 import FullBodyReactiveRoleMovementPlan
from rosclaw_soccer.training.reactive_route_growth import (
    build_reactive_movement_plan,
    default_reactive_development_cases,
    default_reactive_retention_manifest,
    default_reactive_retention_manifest_v2,
    label_reactive_route_failure,
)


def _samples() -> tuple[ReactiveRouteSample, ...]:
    rng = np.random.default_rng(122)
    weights = rng.normal(0.0, 0.08, size=(2, len(REACTIVE_ROUTE_FEATURE_NAMES)))
    rows = []
    for index in range(1_024):
        features = rng.normal(0.0, 1.0, size=len(REACTIVE_ROUTE_FEATURE_NAMES))
        features[10:14] = (index % 2, (index + 1) % 2, index % 2, (index + 1) % 2)
        command = weights @ features
        rows.append(
            ReactiveRouteSample(
                episode_id=f"episode-{index % 8}",
                features=tuple(float(value) for value in features),
                teacher_world_command_xy_mps=(float(command[0]), float(command[1])),
            )
        )
    return tuple(rows)


def test_reactive_route_features_are_relative_and_role_conditioned() -> None:
    features = reactive_route_features(
        target_xy_m=np.asarray((3.0, 2.0)),
        self_position_xy_m=np.asarray((1.0, 0.5)),
        self_velocity_xy_mps=np.asarray((0.2, -0.1)),
        ball_position_xy_m=np.asarray((1.5, 0.8)),
        carrier_position_xy_m=np.asarray((0.0, 0.0)),
        other_role_position_xy_m=np.asarray((2.0, -0.5)),
        action="pass",
        role="teammate",
    )
    assert features.tolist() == pytest.approx(
        [2.0, 1.5, 0.2, -0.1, 0.5, 0.3, -1.0, -0.5, 1.0, -1.0, 1, 0, 1, 0]
    )


def test_reactive_route_actor_round_trip_and_support_fallback(tmp_path: Path) -> None:
    actor = fit_reactive_route_actor(
        _samples(),
        source_stage_hash=hash_json({"stage": "s121"}),
    )
    path = tmp_path / "actor.json"
    save_reactive_route_actor(actor, path)
    loaded = load_reactive_route_actor(path)
    assert loaded == actor
    assert loaded.decide(np.zeros(len(REACTIVE_ROUTE_FEATURE_NAMES))).accepted
    rejected = loaded.decide(np.full(len(REACTIVE_ROUTE_FEATURE_NAMES), 100.0))
    assert not rejected.accepted
    assert rejected.world_command_xy_mps == (0.0, 0.0)

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["output_weights"][0][0] += 0.1
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="hash does not match"):
        load_reactive_route_actor(path)

    save_reactive_route_actor(actor, path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["output_authority"] = "DIRECT_JOINT_TORQUE"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="metadata contract"):
        load_reactive_route_actor(path)


def test_reactive_route_actor_is_permanently_sim_only() -> None:
    actor = fit_reactive_route_actor(
        _samples(),
        source_stage_hash=hash_json({"stage": "s121"}),
    )
    payload = actor.to_dict(include_hash=False)
    payload["hardware_authorized"] = True
    for key in (
        "feature_names",
        "algorithm",
        "pixels_used_for_training",
        "current_stage_retention_evidence_used_for_training",
        "released_source_stage_evidence_used_for_training",
        "output_authority",
    ):
        payload.pop(key)
    payload["feature_center"] = tuple(payload["feature_center"])
    payload["feature_scale"] = tuple(payload["feature_scale"])
    payload["feature_minimum"] = tuple(payload["feature_minimum"])
    payload["feature_maximum"] = tuple(payload["feature_maximum"])
    payload["output_weights"] = tuple(tuple(row) for row in payload["output_weights"])
    payload["training_episode_ids"] = tuple(payload["training_episode_ids"])
    with pytest.raises(ValueError, match="SIM-only contract"):
        type(actor)(**payload)


def test_reactive_role_plan_binds_actor_and_has_no_hardware_authority(tmp_path: Path) -> None:
    actor = fit_reactive_route_actor(
        _samples(),
        source_stage_hash=hash_json({"stage": "s121"}),
    )
    path = tmp_path / "actor.json"
    save_reactive_route_actor(actor, path)
    case = default_reactive_development_cases()[0]
    plan = build_reactive_movement_plan(
        scenario=case.scenario,
        action=TacticalAction.PASS,
        actor_path=path,
        actor=actor,
        teammate_origin_offset_m=case.teammate_origin_offset_m,
        defender_origin_offset_m=case.defender_origin_offset_m,
    )
    assert isinstance(plan, FullBodyReactiveRoleMovementPlan)
    assert plan.teammate_movement.role == "teammate"
    assert plan.defender_movement.role == "defender"
    assert plan.teammate_movement.actor_hash == actor.actor_hash
    assert plan.activation_ceiling == "SIM_ONLY"
    assert plan.hardware_authorized is False
    assert plan.plan_hash.startswith("sha256:")


def test_reactive_retention_generations_are_sealed_and_disjoint() -> None:
    first = default_reactive_retention_manifest()
    second = default_reactive_retention_manifest_v2()
    assert first.training_access_allowed is second.training_access_allowed is False
    assert first.activation_ceiling == second.activation_ceiling == "SIM_ONLY"
    assert {case.case_hash for case in first.cases}.isdisjoint(
        case.case_hash for case in second.cases
    )


def test_failure_relabel_is_bounded_and_keeps_both_roles() -> None:
    frames = 10
    base = np.zeros((frames, len(REACTIVE_ROUTE_FEATURE_NAMES)), dtype=np.float64)
    base[:, :2] = (0.6, -0.5)
    trajectory = {
        "passer_reactive_route_features": base.copy(),
        "goalkeeper_reactive_route_features": base.copy(),
    }
    rows = label_reactive_route_failure(
        trajectory,
        episode_id="failure-001",
        teammate_lateral_bias_m=-0.08,
    )
    assert len(rows) == 2 * frames
    assert {row.episode_id for row in rows} == {
        "failure-001.passer",
        "failure-001.goalkeeper",
    }
    assert max(np.linalg.norm(row.teacher_world_command_xy_mps) for row in rows) <= 0.45
    with pytest.raises(ValueError, match="lateral bias"):
        label_reactive_route_failure(
            trajectory,
            episode_id="failure-002",
            teammate_lateral_bias_m=-0.2,
        )
