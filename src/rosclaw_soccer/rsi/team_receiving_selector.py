"""Measured-state offline selector for historical eight-body SIM research.

This module returns an index only. The caller must enforce existing physical
ownership, protection, motor and promotion gates; no hardware authority.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TeamReceivingSelector:
    center: np.ndarray
    scale: np.ndarray
    weight1: np.ndarray
    bias1: np.ndarray
    weight2: np.ndarray
    bias2: np.ndarray
    offsets: np.ndarray

    @classmethod
    def from_dict(cls, value: dict) -> "TeamReceivingSelector":
        if (
            not isinstance(value, dict)
            or value.get("schema") != "soccer.team_receiving_offline_actor.v1"
            or value.get("hardware_authorized") is not False
            or value.get("promotion_authorized") is not False
        ):
            raise ValueError("unpromoted SIM_ONLY selector required")
        fields = ("center", "scale", "weight1", "bias1", "weight2", "bias2", "offsets")
        arrays = [np.asarray(value.get(name), dtype=np.float64) for name in fields]
        shapes = ((10,), (10,), (32, 10), (32,), (16, 32), (16,), (16, 12))
        if any(
            a.shape != shape or not np.isfinite(a).all()
            for a, shape in zip(arrays, shapes, strict=True)
        ):
            raise ValueError("complete finite bounded selector parameters required")
        if np.any(arrays[1] < 0.01) or np.max(np.abs(arrays[6])) > 1 or np.any(arrays[6][0] != 0):
            raise ValueError("bounded action offsets and nonzero scaling required")
        immutable = [a.copy() for a in arrays]
        for array in immutable:
            array.flags.writeable = False
        return cls(*immutable)

    def select(self, features: tuple[float, ...]) -> tuple[int, bool]:
        values = np.asarray(features, dtype=np.float64)
        if values.shape != (10,) or not np.isfinite(values).all():
            raise ValueError("finite same-frame measured receiving features required")
        normalized = (values - self.center) / self.scale
        if np.max(np.abs(normalized)) > 5.0:
            return 0, True
        hidden = np.tanh(self.weight1 @ normalized + self.bias1)
        logits = self.weight2 @ hidden + self.bias2
        if not np.isfinite(logits).all():
            raise ValueError("nonfinite actor output")
        return int(np.argmax(logits)), False
