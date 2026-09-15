"""A bounded local action subspace; no motion authority or universal-prior claim."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

import numpy as np

from rosclaw_soccer.sim.contracts import hash_json


@dataclass(frozen=True)
class MotorSynergyBasis:
    matrix: tuple[tuple[float, ...], ...]
    demonstration_hash: str
    raw_amplitude: float = 0.2

    def __post_init__(self) -> None:
        if (
            type(self.matrix) is not tuple
            or not 3 <= len(self.matrix) <= 256
            or any(type(row) is not tuple or len(row) != 3 for row in self.matrix)
            or any(
                type(x) not in (int, float) or not math.isfinite(x)
                for row in self.matrix
                for x in row
            )
            or type(self.raw_amplitude) not in (int, float)
            or not math.isfinite(self.raw_amplitude)
            or not 0 < self.raw_amplitude <= 0.5
            or not isinstance(self.demonstration_hash, str)
            or re.fullmatch(r"sha256:[0-9a-f]{64}", self.demonstration_hash) is None
        ):
            raise ValueError("finite content-bound three-dimensional motor basis required")
        matrix = np.asarray(self.matrix, dtype=np.float64)
        if not np.allclose(matrix.T @ matrix, np.eye(3), rtol=0, atol=1e-6):
            raise ValueError("orthonormal motor basis columns required")

    @property
    def basis_hash(self) -> str:
        return str(
            hash_json(
                {
                    "schema": "soccer.local_motor_synergy.v1",
                    "matrix": self.matrix,
                    "demonstration_hash": self.demonstration_hash,
                    "raw_amplitude": self.raw_amplitude,
                    "activation_ceiling": "SIM_ONLY",
                    "universal_motion_prior": False,
                }
            )
        )

    def decode(self, raw: np.ndarray) -> np.ndarray:
        self.__post_init__()
        value = np.asarray(raw)
        if value.shape != (3,) or value.dtype.kind != "f" or not np.isfinite(value).all():
            raise ValueError("finite three-dimensional raw latent required")
        return np.asarray(
            np.asarray(self.matrix) @ (self.raw_amplitude * np.tanh(value.astype(np.float64))),
            dtype=np.float32,
        )
