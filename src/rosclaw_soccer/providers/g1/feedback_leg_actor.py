"""Frozen, bounded SIM_ONLY proprioceptive G1 leg feedback policy.

The artifact is trained in isolated MJX/CPU physics. Loading it grants no
team or hardware authority; a shared world must independently qualify each
player and contact outcome before considering promotion.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from rosclaw_soccer.sim.contracts import hash_json
from rosclaw_soccer.skills.team.motor_option import TeamMotorObservation, TeamMotorTarget

JOINT_INDICES = np.asarray((0, 3, 6, 9), dtype=np.int64)
FEATURE_COUNT = 15
MAX_RESIDUAL_RAD = 0.12
ARTIFACT_SCHEMA = "rosclaw_soccer.rsi.mjx_contact_residual_es.v1"


@dataclass(frozen=True)
class FrozenFeedbackLegActor:
    artifact_hash: str
    parameters: NDArray[np.float64]

    def __post_init__(self) -> None:
        if (
            re.fullmatch(r"sha256:[0-9a-f]{64}", self.artifact_hash) is None
            or not isinstance(self.parameters, np.ndarray)
            or self.parameters.shape != (64,)
            or not np.isfinite(self.parameters).all()
        ):
            raise ValueError("finite sealed bounded motor actor required")
        frozen = np.frombuffer(
            np.asarray(self.parameters, dtype=np.float64).tobytes(), dtype=np.float64
        )
        object.__setattr__(self, "parameters", frozen)

    @classmethod
    def load(cls, path: Path) -> FrozenFeedbackLegActor:
        if path.stat().st_size > 1_000_000:
            raise ValueError("motor artifact exceeds size bound")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("sealed motor artifact object required")
        commitment = payload.pop("result_hash", None)
        if (
            commitment != hash_json(payload)
            or payload.get("schema") != ARTIFACT_SCHEMA
            or payload.get("activation_ceiling") != "SIM_ONLY"
            or payload.get("promotion_authorized") is not False
            or payload.get("cpu_holdout_evaluated") is not False
        ):
            raise ValueError("sealed unpromoted SIM_ONLY motor artifact required")
        parameters = np.asarray(payload.get("parameters"), dtype=np.float64)
        if parameters.shape != (64,) or not np.isfinite(parameters).all():
            raise ValueError("finite 64-parameter motor artifact required")
        return cls(str(commitment), parameters)

    def propose(
        self,
        observation: TeamMotorObservation,
        foundation: TeamMotorTarget,
        *,
        foot_seen: bool,
        nonfoot_seen: bool,
    ) -> TeamMotorTarget:
        if not isinstance(observation, TeamMotorObservation) or not isinstance(
            foundation, TeamMotorTarget
        ):
            raise ValueError("typed motor observation and SONIC target required")
        if type(foot_seen) is not bool or type(nonfoot_seen) is not bool:
            raise ValueError("measured contact flags required")
        qpos = np.asarray(observation.qpos, dtype=np.float64)
        qvel = np.asarray(observation.qvel, dtype=np.float64)
        ball_dx = float(qpos[36] - qpos[0])
        features = np.concatenate(
            (
                np.asarray(
                    (
                        ball_dx,
                        qpos[37] - qpos[1],
                        qvel[35] - qvel[0],
                        qvel[0],
                        qpos[2] - 0.75,
                    ),
                    dtype=np.float64,
                ),
                qpos[7 + JOINT_INDICES],
                qvel[6 + JOINT_INDICES] / 5.0,
                np.asarray((foot_seen, nonfoot_seen), dtype=np.float64),
            )
        )
        if features.shape != (FEATURE_COUNT,) or not np.isfinite(features).all():
            raise ValueError("finite body/ball feedback required")
        gate = float((0.05 < ball_dx < 1.2) or foot_seen)
        matrix = self.parameters[: 4 * FEATURE_COUNT].reshape(4, FEATURE_COUNT)
        bias = self.parameters[4 * FEATURE_COUNT :]
        residual = MAX_RESIDUAL_RAD * gate * np.tanh(matrix @ features + bias)
        target = np.asarray(foundation.target_rad, dtype=np.float64).copy()
        target[JOINT_INDICES] += residual
        if not np.isfinite(target).all() or np.any(np.abs(target) > 10):
            raise ValueError("feedback joint target violates team motor contract")
        return TeamMotorTarget(tuple(target.tolist()), foundation.kp, foundation.kd)
