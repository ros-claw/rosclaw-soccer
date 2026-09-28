"""SIM_ONLY causal 500 Hz receiving torque student; no teacher or simulator access."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from rosclaw_soccer.sim.contracts import hash_bytes

FEATURE_COUNT = 77
HIDDEN_COUNT = 64
ACTION_COUNT = 29
MAX_ADDED_TORQUE_NM = 14.0


def receiving_torque_features(
    qpos: NDArray[np.float64],
    qvel: NDArray[np.float64],
    has_foot_contact: bool,
    foot_contact_elapsed_sec: float,
) -> NDArray[np.float32]:
    """Use only current proprioception, current ball state and past tactile memory."""
    if (
        not isinstance(qpos, np.ndarray)
        or not isinstance(qvel, np.ndarray)
        or qpos.shape != (43,)
        or qvel.shape != (41,)
        or not np.isfinite(qpos).all()
        or not np.isfinite(qvel).all()
        or type(has_foot_contact) is not bool
        or not np.isfinite(foot_contact_elapsed_sec)
        or (not has_foot_contact and foot_contact_elapsed_sec != -1.0)
        or (has_foot_contact and foot_contact_elapsed_sec < 0.0)
    ):
        raise ValueError("finite current G1-ball state and causal contact memory required")
    feature = np.concatenate(
        (
            qpos[2:3],
            qpos[3:7],
            qvel[:6],
            qpos[7:36],
            qvel[6:35],
            qpos[36:39] - qpos[:3],
            qvel[35:38] - qvel[:3],
            np.asarray((float(has_foot_contact), min(max(foot_contact_elapsed_sec, 0), 0.2))),
        )
    )
    if feature.shape != (FEATURE_COUNT,):
        raise RuntimeError("receiving student feature schema changed")
    return np.asarray(feature, dtype=np.float32)


@dataclass(frozen=True)
class FrozenReceivingTorqueStudent:
    mean: NDArray[np.float32]
    scale: NDArray[np.float32]
    w1: NDArray[np.float32]
    b1: NDArray[np.float32]
    w2: NDArray[np.float32]
    b2: NDArray[np.float32]
    w3: NDArray[np.float32]
    b3: NDArray[np.float32]

    def __post_init__(self) -> None:
        expected = {
            "mean": (FEATURE_COUNT,),
            "scale": (FEATURE_COUNT,),
            "w1": (FEATURE_COUNT, HIDDEN_COUNT),
            "b1": (HIDDEN_COUNT,),
            "w2": (HIDDEN_COUNT, HIDDEN_COUNT),
            "b2": (HIDDEN_COUNT,),
            "w3": (HIDDEN_COUNT, ACTION_COUNT),
            "b3": (ACTION_COUNT,),
        }
        for name, shape in expected.items():
            value = getattr(self, name)
            if (
                not isinstance(value, np.ndarray)
                or value.shape != shape
                or not np.isfinite(value).all()
                or np.max(np.abs(value)) > 1e4
            ):
                raise ValueError(f"finite bounded receiving student {name} required")
            immutable = np.frombuffer(
                np.asarray(value, dtype=np.float32).tobytes(), dtype=np.float32
            )
            object.__setattr__(self, name, immutable.reshape(shape))
        if np.any(self.scale < 0.05):
            raise ValueError("receiving student normalization scale below floor")

    @classmethod
    def load_npz(cls, path: Path, *, expected_hash: str) -> FrozenReceivingTorqueStudent:
        if path.stat().st_size > 5_000_000 or hash_bytes(path.read_bytes()) != expected_hash:
            raise ValueError("sealed bounded receiving student artifact required")
        with np.load(path, allow_pickle=False) as payload:
            if set(payload.files) != {"mean", "scale", "w1", "b1", "w2", "b2", "w3", "b3"}:
                raise ValueError("complete receiving student parameter set required")
            values = {key: np.asarray(payload[key], dtype=np.float32) for key in payload.files}
        return cls(**values)

    def predict(
        self,
        qpos: NDArray[np.float64],
        qvel: NDArray[np.float64],
        has_foot_contact: bool,
        foot_contact_elapsed_sec: float,
    ) -> NDArray[np.float64]:
        feature = receiving_torque_features(qpos, qvel, has_foot_contact, foot_contact_elapsed_sec)
        x = np.clip((feature - self.mean) / self.scale, -10.0, 10.0)
        x = np.maximum(x @ self.w1 + self.b1, 0.0)
        x = np.maximum(x @ self.w2 + self.b2, 0.0)
        action = MAX_ADDED_TORQUE_NM * np.tanh(x @ self.w3 + self.b3)
        return np.asarray(action, dtype=np.float64)
