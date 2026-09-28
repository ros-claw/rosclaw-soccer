"""Frozen SIM_ONLY ten-joint receiving feedback with sealed evidence admission.

This adapter maps current proprioception and ball state to a bounded joint
target proposal. It owns neither a simulator nor an actuator. Loading requires
independent training and holdout evidence; it never authorizes promotion.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.skills.team.motor_option import TeamMotorObservation, TeamMotorTarget

JOINT_INDICES = np.asarray((0, 1, 3, 4, 5, 6, 7, 9, 13, 14), dtype=np.int64)
FEATURE_COUNT = 5 + 2 * len(JOINT_INDICES)
WEIGHT_COUNT = len(JOINT_INDICES) * (FEATURE_COUNT + 1)
MAX_RESIDUAL_RAD = 0.12
ARTIFACT_SCHEMA = "rosclaw_soccer.rsi.receiving_body_actor.v1"
_HASH = re.compile(r"sha256:[0-9a-f]{64}")


@dataclass(frozen=True)
class FrozenReceivingBodyActor:
    artifact_hash: str
    parameters: NDArray[np.float32]

    def __post_init__(self) -> None:
        if (
            _HASH.fullmatch(self.artifact_hash) is None
            or not isinstance(self.parameters, np.ndarray)
            or self.parameters.shape != (WEIGHT_COUNT,)
            or not np.isfinite(self.parameters).all()
        ):
            raise ValueError("finite sealed ten-joint receiving actor required")
        immutable = np.frombuffer(
            np.asarray(self.parameters, dtype=np.float32).tobytes(), dtype=np.float32
        )
        object.__setattr__(self, "parameters", immutable)

    @classmethod
    def load(cls, path: Path) -> FrozenReceivingBodyActor:
        if path.stat().st_size > 1_000_000:
            raise ValueError("receiving actor artifact exceeds size bound")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("sealed receiving artifact object required")
        commitment = payload.pop("artifact_hash", None)
        values = np.asarray(payload.get("full_weights"), dtype=np.float32)
        training_path = payload.get("training_report_path")
        fresh_path = payload.get("fresh8_report_path")
        if (
            commitment != hash_json(payload)
            or payload.get("schema") != ARTIFACT_SCHEMA
            or payload.get("activation_ceiling") != "SIM_ONLY"
            or payload.get("promotion_authorized") is not False
            or payload.get("training_gate_passed") is not True
            or payload.get("fresh8_gate_passed") is not True
            or payload.get("shared_world_qualified") is not False
            or _HASH.fullmatch(str(payload.get("training_report_hash"))) is None
            or _HASH.fullmatch(str(payload.get("fresh8_report_hash"))) is None
            or type(training_path) is not str
            or type(fresh_path) is not str
            or not Path(training_path).is_absolute()
            or not Path(fresh_path).is_absolute()
            or values.shape != (WEIGHT_COUNT,)
            or not np.isfinite(values).all()
            or payload.get("full_weights_hash") != hash_bytes(values.tobytes())
        ):
            raise ValueError("training-and-Fresh8-qualified unpromoted actor required")
        training = _sealed_report(
            Path(training_path),
            expected_hash=str(payload["training_report_hash"]),
            schema="rosclaw_soccer.rsi.receiving_body_training.v1",
        )
        fresh = _sealed_report(
            Path(fresh_path),
            expected_hash=str(payload["fresh8_report_hash"]),
            schema="rosclaw_soccer.rsi.receiving_body_fresh8_exam.v1",
        )
        if (
            training.get("training_gate_passed") is not True
            or training.get("full_weights_hash") != payload["full_weights_hash"]
            or fresh.get("fresh8_gate_passed") is not True
            or fresh.get("training_report_hash") != payload["training_report_hash"]
            or fresh.get("full_weights_hash") != payload["full_weights_hash"]
            or fresh.get("compiled_model_hash") != training.get("compiled_model_hash")
            or _HASH.fullmatch(str(training.get("compiled_model_hash"))) is None
        ):
            raise ValueError("bound CPU training and Fresh8 physical evidence required")
        return cls(str(commitment), values)

    def propose(
        self, observation: TeamMotorObservation, foundation: TeamMotorTarget
    ) -> TeamMotorTarget:
        if not isinstance(observation, TeamMotorObservation) or not isinstance(
            foundation, TeamMotorTarget
        ):
            raise ValueError("typed measured motor observation and foundation required")
        qpos = np.asarray(observation.qpos, dtype=np.float64)
        qvel = np.asarray(observation.qvel, dtype=np.float64)
        dx = float(qpos[36] - qpos[0])
        if not 0.12 < dx < 0.85:
            return foundation
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
        matrix = self.parameters[: len(JOINT_INDICES) * FEATURE_COUNT].reshape(
            len(JOINT_INDICES), FEATURE_COUNT
        )
        bias = self.parameters[len(JOINT_INDICES) * FEATURE_COUNT :]
        residual = MAX_RESIDUAL_RAD * np.tanh(matrix @ feature + bias)
        target = np.asarray(foundation.target_rad, dtype=np.float64).copy()
        target[JOINT_INDICES] += residual
        if not np.isfinite(target).all() or np.any(np.abs(target) > 10):
            raise ValueError("receiving feedback joint target violates motor contract")
        return TeamMotorTarget(tuple(target.tolist()), foundation.kp, foundation.kd)


def _sealed_report(path: Path, *, expected_hash: str, schema: str) -> dict[str, object]:
    if path.stat().st_size > 1_000_000:
        raise ValueError("physical evidence report exceeds size bound")
    report = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(report, dict):
        raise ValueError("typed physical evidence report required")
    commitment = report.pop("report_hash", None)
    if (
        commitment != expected_hash
        or commitment != hash_json(report)
        or report.get("schema") != schema
        or report.get("activation_ceiling") != "SIM_ONLY"
        or report.get("promotion_authorized") is not False
    ):
        raise ValueError("sealed SIM_ONLY physical evidence required")
    return report
