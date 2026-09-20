"""Bounded SONIC token proposals for SIM_ONLY reachability experiments.

The frozen encoder supplies the nominal motion token. Perturbations are not
assumed to remain on its learned manifold; physics still decides validity.
This is not an arbitrary token replacement, weight update or torque interface.
"""

import math
from dataclasses import asdict, dataclass

import numpy as np

from rosclaw_soccer.sim.contracts import hash_json


@dataclass(frozen=True)
class SonicLatentSchedule:
    knots: tuple[tuple[float, ...], ...]
    knot_frames: int = 20
    relative_l2_limit: float = 0.1
    absolute_l2_limit: float = 0.5
    ramp_frames: int = 10

    def __post_init__(self) -> None:
        if (
            type(self.knots) is not tuple
            or not 1 <= len(self.knots) <= 32
            or type(self.knot_frames) is not int
            or not 1 <= self.knot_frames <= 100
            or type(self.ramp_frames) is not int
            or self.ramp_frames != 10
            or type(self.relative_l2_limit) is not float
            or self.relative_l2_limit != 0.1
            or type(self.absolute_l2_limit) is not float
            or self.absolute_l2_limit != 0.5
            or any(
                type(row) is not tuple
                or len(row) != 64
                or any(
                    type(v) not in (int, float) or not math.isfinite(v) or abs(v) > 1 for v in row
                )
                for row in self.knots
            )
        ):
            raise ValueError("bounded 64D SONIC token knots and fixed envelope required")

    @property
    def contract_hash(self) -> str:
        return str(hash_json({"schema": "soccer.sonic_latent_schedule.v1", **asdict(self)}))

    def transform(self, encoded: np.ndarray, frame: int) -> np.ndarray:
        if (
            type(frame) is not int
            or frame < 0
            or encoded.shape != (1, 64)
            or encoded.dtype != np.float32
            or not np.isfinite(encoded).all()
        ):
            raise ValueError("finite float32 encoder token and local frame required")
        position = frame / self.knot_frames
        left = min(int(position), len(self.knots) - 1)
        right = min(left + 1, len(self.knots) - 1)
        fraction = min(position - left, 1.0)
        proposed = (1 - fraction) * np.asarray(self.knots[left]) + fraction * np.asarray(
            self.knots[right]
        )
        radius = min(
            self.absolute_l2_limit, self.relative_l2_limit * float(np.linalg.norm(encoded))
        )
        ramp = min(1.0, (frame + 1) / self.ramp_frames)
        delta = proposed * (radius * ramp / math.sqrt(64))
        if not np.any(delta):
            return encoded.copy()
        result = encoded + delta.astype(np.float32)[None, :]
        if not np.isfinite(result).all():
            raise ValueError("nonfinite transformed token")
        return result
