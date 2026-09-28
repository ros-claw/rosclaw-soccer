"""Sealed, bounded receiving-body feedback actor boundary tests."""

import json
from pathlib import Path

import numpy as np
import pytest

from rosclaw_soccer.providers.g1.receiving_body_actor import (
    ARTIFACT_SCHEMA,
    WEIGHT_COUNT,
    FrozenReceivingBodyActor,
)
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.skills.team.motor_option import TeamMotorObservation, TeamMotorTarget


def _artifact(path: Path, *, trained: bool = True, fresh: bool = True) -> Path:
    weights = np.zeros(WEIGHT_COUNT, dtype=np.float32)
    weights[-10] = 1.0
    model_hash = "sha256:" + "3" * 64
    training = {
        "schema": "rosclaw_soccer.rsi.receiving_body_training.v1",
        "activation_ceiling": "SIM_ONLY",
        "promotion_authorized": False,
        "training_gate_passed": trained,
        "full_weights_hash": hash_bytes(weights.tobytes()),
        "compiled_model_hash": model_hash,
    }
    training["report_hash"] = hash_json(training)
    training_path = path.with_name("training.json")
    training_path.write_text(json.dumps(training), encoding="utf-8")
    fresh_report = {
        "schema": "rosclaw_soccer.rsi.receiving_body_fresh8_exam.v1",
        "activation_ceiling": "SIM_ONLY",
        "promotion_authorized": False,
        "fresh8_gate_passed": fresh,
        "training_report_hash": training["report_hash"],
        "full_weights_hash": hash_bytes(weights.tobytes()),
        "compiled_model_hash": model_hash,
    }
    fresh_report["report_hash"] = hash_json(fresh_report)
    fresh_path = path.with_name("fresh.json")
    fresh_path.write_text(json.dumps(fresh_report), encoding="utf-8")
    payload = {
        "schema": ARTIFACT_SCHEMA,
        "activation_ceiling": "SIM_ONLY",
        "promotion_authorized": False,
        "training_gate_passed": trained,
        "fresh8_gate_passed": fresh,
        "shared_world_qualified": False,
        "training_report_hash": training["report_hash"],
        "fresh8_report_hash": fresh_report["report_hash"],
        "training_report_path": str(training_path),
        "fresh8_report_path": str(fresh_path),
        "full_weights": weights.tolist(),
        "full_weights_hash": hash_bytes(weights.tobytes()),
    }
    payload["artifact_hash"] = hash_json(payload)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _observation(dx: float) -> TeamMotorObservation:
    qpos = np.zeros(43)
    qpos[2] = 0.75
    qpos[36] = dx
    return TeamMotorObservation(
        agent_id="red.finisher",
        frame=0,
        time_sec=0.0,
        intent="other",
        prospective_owner=False,
        qpos=tuple(qpos.tolist()),
        qvel=tuple(np.zeros(41).tolist()),
        target_position_m=(1.0, 0.0, 0.0),
    )


def _foundation() -> TeamMotorTarget:
    return TeamMotorTarget((0.0,) * 29, (100.0,) * 29, (2.0,) * 29)


def test_qualified_artifact_is_bounded_and_outside_gate_exact(tmp_path: Path) -> None:
    actor = FrozenReceivingBodyActor.load(_artifact(tmp_path / "body.json"))
    foundation = _foundation()
    assert actor.propose(_observation(1.0), foundation) is foundation
    proposal = actor.propose(_observation(0.5), foundation)
    assert 0 < proposal.target_rad[0] <= 0.12
    assert proposal.target_rad[1:] == foundation.target_rad[1:]
    assert proposal.kp == foundation.kp
    assert proposal.kd == foundation.kd
    with pytest.raises(ValueError):
        actor.parameters[0] = 5.0


@pytest.mark.parametrize("trained,fresh", [(False, True), (True, False)])
def test_unqualified_artifact_never_loads(tmp_path: Path, trained: bool, fresh: bool) -> None:
    path = _artifact(tmp_path / "unqualified.json", trained=trained, fresh=fresh)
    with pytest.raises(ValueError, match="training-and-Fresh8"):
        FrozenReceivingBodyActor.load(path)


def test_tampered_artifact_never_loads(tmp_path: Path) -> None:
    path = _artifact(tmp_path / "body.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["full_weights"][0] = 2.0
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="training-and-Fresh8"):
        FrozenReceivingBodyActor.load(path)


def test_tampered_fresh_evidence_never_loads(tmp_path: Path) -> None:
    path = _artifact(tmp_path / "body.json")
    fresh_path = tmp_path / "fresh.json"
    report = json.loads(fresh_path.read_text(encoding="utf-8"))
    report["fresh8_gate_passed"] = False
    fresh_path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="sealed SIM_ONLY"):
        FrozenReceivingBodyActor.load(path)
