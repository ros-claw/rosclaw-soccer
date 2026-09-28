"""Pure SIM_ONLY receiving torque student and teacher-exclusion checks."""

from pathlib import Path

import numpy as np
import pytest
from rsi_receiving_contact_dynamics_audit import _run

from rosclaw_soccer.providers.g1.receiving_torque_student import (
    ACTION_COUNT,
    FEATURE_COUNT,
    HIDDEN_COUNT,
    FrozenReceivingTorqueStudent,
    receiving_torque_features,
)
from rosclaw_soccer.sim.contracts import hash_bytes


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


def _state() -> tuple[np.ndarray, np.ndarray]:
    qpos = np.zeros(43, dtype=np.float64)
    qpos[2] = 0.75
    qpos[3] = 1.0
    qpos[36:39] = (0.3, 0.1, 0.115)
    return qpos, np.zeros(41, dtype=np.float64)


def test_student_is_causal_bounded_and_immutable() -> None:
    student = _student()
    qpos, qvel = _state()
    feature = receiving_torque_features(qpos, qvel, False, -1.0)
    assert feature.shape == (FEATURE_COUNT,)
    assert np.array_equal(student.predict(qpos, qvel, False, -1.0), np.zeros(ACTION_COUNT))
    with pytest.raises(ValueError):
        student.w1[0, 0] = 4.0
    qpos[36] += 0.05
    assert receiving_torque_features(qpos, qvel, False, -1.0)[-8] != feature[-8]


@pytest.mark.parametrize("elapsed,has_foot", [(0.0, False), (-1.0, True)])
def test_contact_memory_must_be_causal(elapsed: float, has_foot: bool) -> None:
    qpos, qvel = _state()
    with pytest.raises(ValueError, match="causal contact"):
        receiving_torque_features(qpos, qvel, has_foot, elapsed)


def test_typed_npz_load_rejects_tampering(tmp_path: Path) -> None:
    student = _student()
    path = tmp_path / "student.npz"
    np.savez_compressed(
        path, **{key: getattr(student, key) for key in student.__dataclass_fields__}
    )
    valid_hash = hash_bytes(path.read_bytes())
    loaded = FrozenReceivingTorqueStudent.load_npz(path, expected_hash=valid_hash)
    qpos, qvel = _state()
    assert np.array_equal(loaded.predict(qpos, qvel, False, -1.0), np.zeros(ACTION_COUNT))
    with pytest.raises(ValueError, match="sealed bounded"):
        FrozenReceivingTorqueStudent.load_npz(path, expected_hash="sha256:" + "0" * 64)


def test_student_cannot_stack_on_privileged_teacher() -> None:
    with pytest.raises(ValueError, match="must replace"):
        _run(  # type: ignore[arg-type]
            None,
            {},
            np.zeros(80),
            privileged_teacher_lateral_sign=1.0,
            actor_torque_fn=lambda *_: np.zeros(ACTION_COUNT),
        )
