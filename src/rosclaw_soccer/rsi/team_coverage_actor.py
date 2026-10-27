"""SIM-only 17-way actor with an explicit unchanged-parent fallback."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TeamCoverageActor:
    center: np.ndarray
    scale: np.ndarray
    weight1: np.ndarray
    bias1: np.ndarray
    weight2: np.ndarray
    bias2: np.ndarray
    support: np.ndarray
    allowed: np.ndarray
    offsets: np.ndarray
    radius: float

    @classmethod
    def from_dict(cls, value: dict) -> "TeamCoverageActor":
        if (
            value.get("schema") != "soccer.team_coverage_actor.v1"
            or value.get("hardware_authorized") is not False
            or value.get("promotion_authorized") is not False
        ):
            raise ValueError("unpromoted SIM_ONLY actor required")
        names = (
            "center",
            "scale",
            "weight1",
            "bias1",
            "weight2",
            "bias2",
            "support",
            "allowed",
            "offsets",
        )
        if np.asarray(value["allowed"]).dtype != np.bool_:
            raise ValueError("explicit boolean safety mask required")
        arrays = [
            np.asarray(value[name], dtype=bool if name == "allowed" else np.float64)
            for name in names
        ]
        shapes = ((10,), (10,), (32, 10), (32,), (17, 32), (17,), (6, 10), (6, 16), (16, 12))
        if any(
            a.shape != shape or not np.isfinite(a).all()
            for a, shape in zip(arrays, shapes, strict=True)
        ):
            raise ValueError("finite complete model parameters required")
        radius = value.get("radius")
        if (
            type(radius) is not float
            or not 0 < radius <= 0.004
            or np.any(arrays[1] < 0.01)
            or np.max(np.abs(arrays[8])) > 1
            or np.any(arrays[8][0] != 0)
        ):
            raise ValueError("bounded support and action offsets required")
        arrays = [a.copy() for a in arrays]
        for a in arrays:
            a.flags.writeable = False
        return cls(*arrays, radius)

    def select(self, features: tuple[float, ...]) -> tuple[int, bool]:
        x = np.asarray(features, dtype=np.float64)
        if x.shape != (10,) or not np.isfinite(x).all():
            raise ValueError("finite measured state required")
        distances = np.linalg.norm(self.support - x, axis=1)
        index = int(np.argmin(distances))
        if distances[index] > self.radius:
            return 16, True
        hidden = np.tanh(self.weight1 @ ((x - self.center) / self.scale) + self.bias1)
        logits = self.weight2 @ hidden + self.bias2
        if not np.isfinite(logits).all():
            raise ValueError("nonfinite actor output")
        action = int(np.argmax(logits))
        if action != 16 and not self.allowed[index, action]:
            return 16, True
        return action, False
