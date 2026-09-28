"""Evidence-bound SIM_ONLY research bridge for a teacher-free receiving student.

This is not a runtime or hardware adapter. It carries the frozen student plus
the exact ten-joint pre-contact bias used in its CPU training and Fresh8 exam.
The bundle grants only a shared-world simulation experiment, not promotion.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from rosclaw_soccer.providers.g1.receiving_torque_student import FrozenReceivingTorqueStudent
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json

# Frozen 10-joint bias schema from the source training artifact. Keep this
# module import-independent of the team package to avoid a simulator cycle.
JOINT_INDICES = np.asarray((0, 1, 3, 4, 5, 6, 7, 9, 13, 14), dtype=np.int64)
BIAS_FEATURE_COUNT = 5 + 2 * len(JOINT_INDICES)
WEIGHT_COUNT = len(JOINT_INDICES) * (BIAS_FEATURE_COUNT + 1)
MAX_RESIDUAL_RAD = 0.12


def _read_report(path: Path, *, schema: str) -> tuple[dict[str, object], str]:
    if not path.is_absolute() or path.stat().st_size > 2_000_000:
        raise ValueError("bounded absolute SIM_ONLY evidence report required")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("typed SIM_ONLY evidence report required")
    commitment = value.pop("report_hash", None)
    if (
        commitment != hash_json(value)
        or value.get("schema") != schema
        or value.get("activation_ceiling") != "SIM_ONLY"
        or value.get("promotion_authorized") is not False
    ):
        raise ValueError("sealed unpromoted SIM_ONLY evidence report required")
    return value, str(commitment)


@dataclass(frozen=True)
class QualifiedReceivingStudent:
    """Frozen learned torque plus frozen source bias, scoped to simulation."""

    student: FrozenReceivingTorqueStudent
    bias_parameters: NDArray[np.float32]
    training_report_hash: str
    fresh_report_hash: str
    model_hash: str
    activation_ceiling: str = "SIM_ONLY"

    def __post_init__(self) -> None:
        if (
            not isinstance(self.student, FrozenReceivingTorqueStudent)
            or not isinstance(self.bias_parameters, np.ndarray)
            or self.bias_parameters.shape != (WEIGHT_COUNT,)
            or not np.isfinite(self.bias_parameters).all()
            or self.activation_ceiling != "SIM_ONLY"
            or any(
                not isinstance(value, str) or not value.startswith("sha256:") or len(value) != 71
                for value in (self.training_report_hash, self.fresh_report_hash, self.model_hash)
            )
        ):
            raise ValueError("finite evidence-bound SIM_ONLY receiving student required")
        immutable = np.frombuffer(
            np.asarray(self.bias_parameters, dtype=np.float32).tobytes(), dtype=np.float32
        )
        object.__setattr__(self, "bias_parameters", immutable)

    @classmethod
    def load(
        cls,
        *,
        warm_start: Path,
        training: Path,
        fresh: Path,
    ) -> QualifiedReceivingStudent:
        warm, warm_hash = _read_report(
            warm_start, schema="rosclaw_soccer.rsi.cpu_receiving_bias_trust_region.v1"
        )
        train, train_hash = _read_report(
            training, schema="rosclaw_soccer.rsi.cpu_receiving_torque_distill.v1"
        )
        exam, exam_hash = _read_report(
            fresh, schema="rosclaw_soccer.rsi.cpu_receiving_torque_fresh8.v1"
        )
        parameters = np.asarray(warm.get("full_weights"), dtype=np.float32)
        if (
            train.get("warm_start_hash") != warm_hash
            or train.get("training_gate_passed") is not True
            or train.get("teacher_used_during_student_exam") is not False
            or train.get("fresh8_opened") is not False
            or exam.get("training_report_hash") != train_hash
            or exam.get("fresh8_gate_passed") is not True
            or exam.get("teacher_used") is not False
            or exam.get("shared_world_qualified") is not False
            or exam.get("model_hash") != train.get("model_hash")
            or exam.get("compiled_model_hash") != train.get("compiled_model_hash")
            or parameters.shape != (WEIGHT_COUNT,)
            or not np.isfinite(parameters).all()
            or warm.get("full_weights_hash") != hash_bytes(parameters.tobytes())
        ):
            raise ValueError("training-and-Fresh8-qualified unpromoted receiving bundle required")
        student = FrozenReceivingTorqueStudent.load_npz(
            training.parent / "student.npz", expected_hash=str(train["model_hash"])
        )
        return cls(student, parameters, train_hash, exam_hash, str(train["model_hash"]))

    def motor_target(
        self,
        *,
        qpos: NDArray[np.float64],
        qvel: NDArray[np.float64],
        foundation_target: NDArray[np.float64],
        joint_ranges: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        if (
            qpos.shape != (43,)
            or qvel.shape != (41,)
            or foundation_target.shape != (29,)
            or joint_ranges.shape != (29, 2)
            or not all(
                np.isfinite(value).all() for value in (qpos, qvel, foundation_target, joint_ranges)
            )
        ):
            raise ValueError("finite local receiving motor observation and limits required")
        target = foundation_target.copy()
        dx = float(qpos[36] - qpos[0])
        if not 0.12 < dx < 0.85:
            return target
        feature = np.concatenate(
            (
                np.asarray(
                    (
                        dx,
                        qpos[37] - qpos[1],
                        qvel[35] - qvel[0],
                        qvel[36] - qvel[1],
                        qpos[2] - 0.75,
                    ),
                    dtype=np.float64,
                ),
                qpos[7 + JOINT_INDICES],
                qvel[6 + JOINT_INDICES] / 5.0,
            )
        )
        matrix = self.bias_parameters[: len(JOINT_INDICES) * BIAS_FEATURE_COUNT].reshape(
            len(JOINT_INDICES), BIAS_FEATURE_COUNT
        )
        bias = self.bias_parameters[len(JOINT_INDICES) * BIAS_FEATURE_COUNT :]
        residual = MAX_RESIDUAL_RAD * np.tanh(matrix @ feature + bias)
        target[JOINT_INDICES] = np.clip(
            target[JOINT_INDICES] + residual,
            joint_ranges[JOINT_INDICES, 0],
            joint_ranges[JOINT_INDICES, 1],
        )
        return target

    def torque(
        self,
        *,
        qpos: NDArray[np.float64],
        qvel: NDArray[np.float64],
        has_foot_contact: bool,
        elapsed_sec: float,
    ) -> NDArray[np.float64]:
        return self.student.predict(qpos, qvel, has_foot_contact, elapsed_sec)
