"""Read-only, caller-owned neural context for SIM composite-teacher research.

Not an actor input extension, motor authorization, or state shared by players.
The native collection host creates it after the same player's neural forward
and before applying that cycle's action. Arrays have immutable byte backing.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np


def sealed_float32(value: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    if (
        not isinstance(value, np.ndarray)
        or value.shape != shape
        or value.dtype != np.float32
        or not np.isfinite(value).all()
    ):
        raise ValueError("exact finite own neural float32 context required")
    return np.frombuffer(value.tobytes(), dtype=np.float32).reshape(shape)


@dataclass(frozen=True)
class CausalMotorContext:
    episode_key: int
    frame: int
    observation: np.ndarray
    h_before: np.ndarray
    c_before: np.ndarray
    h_after: np.ndarray
    c_after: np.ndarray

    def __post_init__(self) -> None:
        if (
            type(self.episode_key) is not int
            or self.episode_key <= 0
            or type(self.frame) is not int
            or not 0 <= self.frame < 300
        ):
            raise ValueError(
                "explicit same-episode identity and bounded actual control frame required"
            )
        for k in ("observation", "h_before", "c_before", "h_after", "c_after"):
            shape = (160,) if k == "observation" else (2, 1, 128)
            object.__setattr__(self, k, sealed_float32(getattr(self, k), shape))

    def assert_current(self, episode: Any) -> None:
        if self.episode_key != id(episode) or self.frame != episode.frame:
            raise ValueError("stale or foreign player's teacher context rejected")
