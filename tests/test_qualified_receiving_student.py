"""SIM_ONLY shared-world receiving bridge arithmetic and evidence checks."""

import json
from pathlib import Path

import numpy as np
import pytest

from rosclaw_soccer.providers.g1.qualified_receiving_student import (
    WEIGHT_COUNT,
    QualifiedReceivingStudent,
)
from rosclaw_soccer.providers.g1.receiving_torque_student import (
    ACTION_COUNT,
    FEATURE_COUNT,
    HIDDEN_COUNT,
    FrozenReceivingTorqueStudent,
)
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json

HASH = "sha256:" + "1" * 64


def _student() -> FrozenReceivingTorqueStudent:
    return FrozenReceivingTorqueStudent(
        mean=np.zeros(FEATURE_COUNT, dtype=np.float32),
        scale=np.ones(FEATURE_COUNT, dtype=np.float32),
        w1=np.zeros((FEATURE_COUNT, HIDDEN_COUNT), dtype=np.float32),
        b1=np.zeros(HIDDEN_COUNT, dtype=np.float32),
        w2=np.zeros((HIDDEN_COUNT, HIDDEN_COUNT), dtype=np.float32),
        b2=np.zeros(HIDDEN_COUNT, dtype=np.float32),
        w3=np.zeros((HIDDEN_COUNT, ACTION_COUNT), dtype=np.float32),
        b3=np.zeros(ACTION_COUNT, dtype=np.float32),
    )


def _report(path: Path, body: dict) -> str:
    body["report_hash"] = hash_json(body)
    path.write_text(json.dumps(body), encoding="utf-8")
    return body["report_hash"]


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    weights = np.zeros(WEIGHT_COUNT, dtype=np.float32)
    weights[-10] = 0.5
    warm = tmp_path / "warm.json"
    warm_hash = _report(
        warm,
        {
            "schema": "rosclaw_soccer.rsi.cpu_receiving_bias_trust_region.v1",
            "activation_ceiling": "SIM_ONLY",
            "promotion_authorized": False,
            "full_weights": weights.tolist(),
            "full_weights_hash": hash_bytes(weights.tobytes()),
        },
    )
    student = _student()
    model = tmp_path / "student.npz"
    np.savez_compressed(
        model, **{key: getattr(student, key) for key in student.__dataclass_fields__}
    )
    model_hash = hash_bytes(model.read_bytes())
    training = tmp_path / "training.json"
    training_hash = _report(
        training,
        {
            "schema": "rosclaw_soccer.rsi.cpu_receiving_torque_distill.v1",
            "activation_ceiling": "SIM_ONLY",
            "promotion_authorized": False,
            "training_gate_passed": True,
            "teacher_used_during_student_exam": False,
            "fresh8_opened": False,
            "warm_start_hash": warm_hash,
            "model_hash": model_hash,
            "compiled_model_hash": HASH,
        },
    )
    fresh = tmp_path / "fresh.json"
    _report(
        fresh,
        {
            "schema": "rosclaw_soccer.rsi.cpu_receiving_torque_fresh8.v1",
            "activation_ceiling": "SIM_ONLY",
            "promotion_authorized": False,
            "training_report_hash": training_hash,
            "fresh8_gate_passed": True,
            "teacher_used": False,
            "shared_world_qualified": False,
            "model_hash": model_hash,
            "compiled_model_hash": HASH,
        },
    )
    return warm, training, fresh


def test_qualified_bridge_loads_and_bounds_target(tmp_path: Path) -> None:
    warm, training, fresh = _fixture(tmp_path)
    bundle = QualifiedReceivingStudent.load(warm_start=warm, training=training, fresh=fresh)
    qpos = np.zeros(43, dtype=np.float64)
    qpos[2] = 0.75
    qpos[3] = 1
    qpos[36] = 0.5
    qvel = np.zeros(41, dtype=np.float64)
    foundation = np.zeros(29, dtype=np.float64)
    joint_ranges = np.tile(np.asarray((-1.0, 1.0)), (29, 1))
    proposal = bundle.motor_target(
        qpos=qpos, qvel=qvel, foundation_target=foundation, joint_ranges=joint_ranges
    )
    assert 0 < proposal[0] < 0.12
    assert np.count_nonzero(proposal) == 1
    assert np.array_equal(
        bundle.torque(qpos=qpos, qvel=qvel, has_foot_contact=False, elapsed_sec=-1), np.zeros(29)
    )
    qpos[36] = 1.0
    assert np.array_equal(
        bundle.motor_target(
            qpos=qpos, qvel=qvel, foundation_target=foundation, joint_ranges=joint_ranges
        ),
        foundation,
    )


def test_tampered_fresh_cannot_load(tmp_path: Path) -> None:
    warm, training, fresh = _fixture(tmp_path)
    payload = json.loads(fresh.read_text(encoding="utf-8"))
    payload["fresh8_gate_passed"] = False
    fresh.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="sealed"):
        QualifiedReceivingStudent.load(warm_start=warm, training=training, fresh=fresh)
