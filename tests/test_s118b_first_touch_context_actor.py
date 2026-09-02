from __future__ import annotations

import json
from dataclasses import replace

import pytest

from rosclaw_soccer.growth.first_touch_context_actor import (
    load_first_touch_context_actor,
    save_first_touch_context_actor,
)
from rosclaw_soccer.growth.first_touch_prototype_actor import (
    load_first_touch_prototype_actor,
    save_first_touch_prototype_actor,
)
from rosclaw_soccer.training.first_touch_context_actor_train import (
    FirstTouchTeacherSample,
    fit_first_touch_context_actor,
    fit_first_touch_prototype_actor,
)
from rosclaw_soccer.training.first_touch_physics import (
    FirstTouchCandidate,
    FirstTouchPhysicsScenario,
)


def _hash(character: str) -> str:
    return "sha256:" + character * 64


def _samples() -> tuple[FirstTouchTeacherSample, ...]:
    contexts = (
        (0.60, -0.05, -30.0, 1.8, True),
        (0.65, 0.00, -22.0, 2.0, True),
        (0.75, 0.02, -15.0, 2.2, True),
        (0.80, 0.05, -10.0, 2.4, True),
        (0.55, -0.10, -35.0, 1.6, False),
        (0.90, 0.10, -5.0, 2.5, False),
    )
    result: list[FirstTouchTeacherSample] = []
    for index, (speed, lateral, direction, outgoing, passed) in enumerate(contexts):
        scenario = FirstTouchPhysicsScenario(
            scenario_id=f"s118b.synthetic.{index}",
            incoming_speed_mps=speed,
            incoming_lateral_m=lateral,
            target_direction_deg=direction,
            target_outgoing_speed_mps=outgoing,
        )
        candidate = FirstTouchCandidate(
            candidate_id=f"teacher.{index}",
            receiver_start_delay_sec=0.15 + 0.05 * speed,
            stance_offset_x=0.10 * (speed - 0.70),
            stance_offset_y=-0.06 + 0.20 * lateral,
            swing_amplitude=0.65 + 0.10 * speed,
            swing_speed_scale=0.80,
            com_shift_y=0.01,
            pelvis_yaw_offset=0.18,
            foot_yaw_offset=0.03,
        )
        result.append(
            FirstTouchTeacherSample(
                report_hash=_hash(chr(ord("a") + index)),
                scenario_hash=scenario.scenario_hash,
                body_hash=_hash("1"),
                kick_prior_hash=_hash("2"),
                source_implementation_hash=_hash("3"),
                scenario=scenario,
                candidate=candidate,
                passed=passed,
                safety_passed=True,
            )
        )
    return tuple(result)


def test_context_actor_fits_bounded_teacher_support_and_rejects_ood(tmp_path) -> None:
    actor = fit_first_touch_context_actor(_samples(), kick_foot="right")
    scenario = FirstTouchPhysicsScenario(
        scenario_id="s118b.synthetic.query",
        incoming_speed_mps=0.70,
        incoming_lateral_m=0.01,
        target_direction_deg=-20.0,
        target_outgoing_speed_mps=2.1,
    )

    decision = actor.decide(scenario, candidate_id="learned.query")

    assert decision.accepted
    assert decision.candidate is not None
    assert decision.candidate.kick_foot == "right"
    assert decision.route == "CONTEXTUAL_RESIDUAL_CANDIDATE"
    assert actor.to_dict()["activation_ceiling"] == "SIM_ONLY"
    assert actor.to_dict()["retention_evidence_used_for_training"] is False

    outside = replace(scenario, incoming_lateral_m=0.15)
    rejected = actor.decide(outside, candidate_id="learned.ood")
    assert not rejected.accepted
    assert rejected.candidate is None
    assert rejected.route == "FROZEN_PARENT_OOD_FALLBACK"

    artifact = tmp_path / "actor.json"
    save_first_touch_context_actor(actor, artifact)
    loaded = load_first_touch_context_actor(artifact)
    assert loaded.actor_hash == actor.actor_hash
    assert loaded.decide(scenario, candidate_id="learned.query") == decision


def test_context_actor_fails_closed_on_retention_leakage() -> None:
    samples = _samples()
    with pytest.raises(ValueError, match="retention scenario leaked"):
        fit_first_touch_context_actor(
            samples,
            kick_foot="right",
            sealed_retention_scenario_hashes=(samples[0].scenario_hash,),
        )


def test_context_actor_artifact_detects_weight_tampering(tmp_path) -> None:
    actor = fit_first_touch_context_actor(_samples(), kick_foot="right")
    artifact = tmp_path / "actor.json"
    save_first_touch_context_actor(actor, artifact)
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    payload["coefficient_matrix"][0][0] += 0.1
    artifact.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="hash does not match"):
        load_first_touch_context_actor(artifact)


def test_prototype_actor_routes_one_intact_contact_mode_and_round_trips(tmp_path) -> None:
    samples = _samples()
    actor = fit_first_touch_prototype_actor(samples, kick_foot="right")
    query = replace(
        samples[1].scenario,
        scenario_id="s118b.synthetic.prototype-query",
        incoming_lateral_m=0.005,
    )

    decision = actor.decide(query, candidate_id="prototype.query")

    assert decision.accepted
    assert decision.candidate is not None
    assert decision.route == "VERIFIED_CONTACT_MODE_PROTOTYPE"
    assert decision.selected_prototype_report_hash in actor.prototype_report_hashes
    selected = actor.prototype_report_hashes.index(decision.selected_prototype_report_hash)
    assert decision.candidate.stance_offset_y == actor.prototype_candidate_vectors[selected][2]
    assert actor.to_dict()["stability_plasticity_contract"]["stability"].startswith("never")

    artifact = tmp_path / "prototype.json"
    save_first_touch_prototype_actor(actor, artifact)
    loaded = load_first_touch_prototype_actor(artifact)
    assert loaded.actor_hash == actor.actor_hash
    assert loaded.decide(query, candidate_id="prototype.query") == decision


def test_prototype_actor_rejects_retention_leakage_and_distant_context() -> None:
    samples = _samples()
    with pytest.raises(ValueError, match="retention scenario leaked"):
        fit_first_touch_prototype_actor(
            samples,
            kick_foot="right",
            sealed_retention_scenario_hashes=(samples[2].scenario_hash,),
        )
    actor = fit_first_touch_prototype_actor(
        samples,
        kick_foot="right",
        maximum_support_distance=0.25,
    )
    distant = FirstTouchPhysicsScenario(
        scenario_id="s118b.synthetic.prototype-ood",
        incoming_speed_mps=1.5,
        incoming_lateral_m=0.18,
        target_direction_deg=40.0,
        target_outgoing_speed_mps=0.5,
    )
    decision = actor.decide(distant, candidate_id="prototype.ood")
    assert not decision.accepted
    assert decision.candidate is None
    assert decision.route == "FROZEN_PARENT_OOD_FALLBACK"


def test_prototype_router_prioritizes_incoming_contact_pocket_over_target() -> None:
    samples = _samples()
    actor = fit_first_touch_prototype_actor(samples, kick_foot="right")
    query = FirstTouchPhysicsScenario(
        scenario_id="s118b.synthetic.contact-pocket-priority",
        incoming_speed_mps=0.65,
        incoming_lateral_m=0.0,
        target_direction_deg=-10.0,
        target_outgoing_speed_mps=2.4,
    )

    decision = actor.decide(query, candidate_id="prototype.pocket-priority")

    assert decision.accepted
    assert decision.selected_prototype_report_hash == samples[1].report_hash
