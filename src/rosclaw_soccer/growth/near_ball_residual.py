"""Private, bounded near-ball leg residuals; NumPy inference, no hardware surface."""

from __future__ import annotations

import math
import re
import zipfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import numpy as np

from rosclaw_soccer.sim.contracts import hash_json

OBSERVATION_DIM = 56
ACTION_DIM = 12
HIDDEN_DIM = 32
RESIDUAL_LIMIT_RAD = 0.10
RESIDUAL_STEP_RAD = 0.02
_SHAPES = {
    "w1": (56, 32),
    "b1": (32,),
    "w2": (32, 12),
    "b2": (12,),
    "wv": (32,),
    "bv": (),
    "log_std": (12,),
}


@dataclass(frozen=True)
class NearBallResidualPolicy:
    agent_ids: tuple[str, ...]
    body_hash: str
    generation: int
    parent_hash: str
    weights: Mapping[str, np.ndarray]

    def __post_init__(self) -> None:
        if (
            len(self.agent_ids) != 8
            or tuple(sorted(set(self.agent_ids))) != self.agent_ids
            or any(re.fullmatch(r"[a-z][a-z0-9_.]{0,80}", x) is None for x in self.agent_ids)
            or any(
                re.fullmatch(r"sha256:[0-9a-f]{64}", h) is None
                for h in (self.body_hash, self.parent_hash)
            )
            or type(self.generation) is not int
            or not 0 <= self.generation <= 1000000
            or set(self.weights) != set(_SHAPES)
        ):
            raise ValueError("invalid private residual policy identity")
        for name, shape in _SHAPES.items():
            value = self.weights[name]
            if (
                value.shape != (8, *shape)
                or not np.all(np.isfinite(value))
                or np.max(np.abs(value)) > 20
            ):
                raise ValueError("invalid finite residual policy weights")
        if np.any(self.weights["log_std"] < -4) or np.any(self.weights["log_std"] > -0.2):
            raise ValueError("residual exploration exceeds its envelope")
        # Bytes-backed arrays cannot be made writeable again by callers.
        frozen = {
            k: np.frombuffer(np.asarray(v, dtype=np.float64).tobytes(), dtype=np.float64).reshape(
                v.shape
            )
            for k, v in self.weights.items()
        }
        object.__setattr__(self, "weights", MappingProxyType(frozen))

    @classmethod
    def initialize(
        cls, agent_ids: tuple[str, ...], body_hash: str, seed: int = 215
    ) -> NearBallResidualPolicy:
        rng = np.random.default_rng(seed)
        weights = {name: np.zeros((8, *shape)) for name, shape in _SHAPES.items()}
        weights["w1"] = rng.normal(0, 0.08, size=(8, 56, 32))
        weights["log_std"].fill(-1.8)
        return cls(agent_ids, body_hash, 0, str(hash_json({"zero_residual": body_hash})), weights)

    @property
    def policy_hash(self) -> str:
        return str(
            hash_json(
                {
                    "agents": self.agent_ids,
                    "body": self.body_hash,
                    "generation": self.generation,
                    "parent": self.parent_hash,
                    "weights": {k: v.tolist() for k, v in self.weights.items()},
                    "limit_rad": RESIDUAL_LIMIT_RAD,
                    "activation_ceiling": "SIM_ONLY",
                }
            )
        )

    def act(
        self, observations: np.ndarray, rng: np.random.Generator, *, explore: bool
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if observations.shape != (8, 56) or not np.all(np.isfinite(observations)):
            raise ValueError("residual observation must be finite 8x56 proprioception")
        w = self.weights
        hidden = np.tanh(np.einsum("ni,nij->nj", observations, w["w1"]) + w["b1"])
        mean = np.einsum("ni,nij->nj", hidden, w["w2"]) + w["b2"]
        sigma = np.exp(w["log_std"])
        latent = mean + sigma * rng.standard_normal(mean.shape) if explore else mean
        log_probability = (
            -0.5 * ((latent - mean) / sigma) ** 2 - w["log_std"] - 0.5 * math.log(2 * math.pi)
        ).sum(axis=1)
        values = (hidden * w["wv"]).sum(axis=1) + w["bv"]
        return latent, log_probability, values

    def save(self, path: Path) -> None:
        if path.suffix != ".npz":
            raise ValueError("residual checkpoint requires an explicit .npz path")
        if path.exists():
            raise FileExistsError(path)
        np.savez_compressed(
            path,
            agent_ids=np.asarray(self.agent_ids),
            body_hash=self.body_hash,
            generation=self.generation,
            parent_hash=self.parent_hash,
            **self.weights,  # type: ignore[arg-type]
        )

    @classmethod
    def load(cls, path: Path) -> NearBallResidualPolicy:
        with zipfile.ZipFile(path) as archive:
            if sum(x.file_size for x in archive.infolist()) > 2_000_000:
                raise ValueError("residual policy archive exceeds size bound")
        with np.load(path, allow_pickle=False) as archive:
            if (
                len(archive.files) != len(set(archive.files))
                or archive["generation"].shape != ()
                or archive["generation"].dtype.kind not in "iu"
            ):
                raise ValueError("residual generation must be an integer scalar")
            if set(archive.files) != set(_SHAPES) | {
                "agent_ids",
                "body_hash",
                "generation",
                "parent_hash",
            }:
                raise ValueError("residual artifact keys differ")
            return cls(
                tuple(archive["agent_ids"].tolist()),
                str(archive["body_hash"]),
                int(archive["generation"]),
                str(archive["parent_hash"]),
                {k: np.asarray(archive[k], dtype=np.float64) for k in _SHAPES},
            )


def bounded_residual(latent: np.ndarray, previous: np.ndarray, active: np.ndarray) -> np.ndarray:
    if (
        latent.shape != (8, 12)
        or previous.shape != (8, 12)
        or active.shape != (8,)
        or active.dtype != np.bool_
        or not np.all(np.isfinite(latent))
        or not np.all(np.isfinite(previous))
        or np.max(np.abs(previous)) > RESIDUAL_LIMIT_RAD
    ):
        raise ValueError("residual target/filter contract violated")
    desired = RESIDUAL_LIMIT_RAD * np.tanh(latent)
    desired[~active] = 0
    return np.asarray(
        previous + np.clip(0.25 * (desired - previous), -RESIDUAL_STEP_RAD, RESIDUAL_STEP_RAD)
    )
