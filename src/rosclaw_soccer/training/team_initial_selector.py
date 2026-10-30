"""Explicit SIM-only selector over two pre-qualified team control packages.

This small supervised model consumes one initial synchronized simulator state.
It proposes a package index only: it is not a motor controller, safety gate,
online RL algorithm, or evidence that either package is safe. Callers must bind
training examples and outcomes to actually executed physical branches.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

_PLAYERS = 8
_FEATURES = 34


def _valid_ids(agent_ids: tuple[str, ...]) -> bool:
    return bool(
        type(agent_ids) is tuple
        and len(agent_ids) == _PLAYERS
        and all(type(agent) is str and bool(agent) for agent in agent_ids)
        and tuple(sorted(set(agent_ids))) == agent_ids
    )


def initial_team_features(
    qpos: NDArray[np.float64], qvel: NDArray[np.float64], *, agent_ids: tuple[str, ...]
) -> NDArray[np.float64]:
    """Ball-relative player position/velocity from a single pre-control state.

    The caller must also bind these canonical identities to the saved model;
    sorting validates order but does not authenticate the roster. Inputs are
    simulator truth, not verified onboard perception.
    No later frame, reward, label, selected action or result is accepted.
    """
    q = np.asarray(qpos, dtype=np.float64)
    v = np.asarray(qvel, dtype=np.float64)
    if (
        not _valid_ids(agent_ids)
        or q.shape != (_PLAYERS, 43)
        or v.shape != (_PLAYERS, 41)
        or not np.isfinite(q).all()
        or not np.isfinite(v).all()
        or not np.array_equal(q[:, 36:43], np.broadcast_to(q[0, 36:43], (8, 7)))
        or not np.array_equal(v[:, 35:41], np.broadcast_to(v[0, 35:41], (8, 6)))
    ):
        raise ValueError("finite synchronized canonical eight-player state required")
    position = q[:, :2] - q[0, 36:38]
    velocity = v[:, :2] - v[0, 35:37]
    features = np.concatenate((position.reshape(-1), velocity.reshape(-1), v[0, 35:37]))
    if features.shape != (_FEATURES,) or not np.isfinite(features).all():
        raise ValueError("invalid causal feature vector")
    return features


def _probability(logit: NDArray[np.float64]) -> NDArray[np.float64]:
    return 1.0 / (1.0 + np.exp(-np.clip(logit, -60.0, 60.0)))


@dataclass(frozen=True)
class InitialTeamSelector:
    """A frozen binary logistic proposal, never an execution authorization."""

    mean: NDArray[np.float64]
    scale: NDArray[np.float64]
    weight: NDArray[np.float64]
    bias: float
    agent_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in ("mean", "scale", "weight"):
            value = np.array(getattr(self, name), dtype=np.float64, copy=True)
            if value.shape != (_FEATURES,) or not np.isfinite(value).all():
                raise ValueError("finite frozen selector parameters required")
            value.flags.writeable = False
            object.__setattr__(self, name, value)
        if np.any(self.scale <= 0) or not np.isfinite(self.bias) or not _valid_ids(self.agent_ids):
            raise ValueError("positive scale and finite bias required")

    def probability(
        self, features: NDArray[np.float64], *, agent_ids: tuple[str, ...]
    ) -> NDArray[np.float64]:
        values = np.asarray(features, dtype=np.float64)
        if (
            agent_ids != self.agent_ids
            or values.ndim != 2
            or values.shape[1] != _FEATURES
            or not np.isfinite(values).all()
        ):
            raise ValueError("finite selector feature matrix required")
        return _probability(((values - self.mean) / self.scale) @ self.weight + self.bias)

    def propose(
        self, features: NDArray[np.float64], *, agent_ids: tuple[str, ...]
    ) -> NDArray[np.int64]:
        """Propose 0 or 1; the caller must validate and authorize execution."""
        return (self.probability(features, agent_ids=agent_ids) >= 0.5).astype(np.int64)

    def save(self, path: Path) -> None:
        """Write a new pickle-free artifact; never overwrite existing evidence."""
        with path.open("xb") as handle:
            np.savez_compressed(
                handle,
                mean=self.mean,
                scale=self.scale,
                weight=self.weight,
                bias=np.asarray(self.bias, dtype=np.float64),
                agent_ids=np.asarray(self.agent_ids, dtype=np.str_),
            )

    @classmethod
    def load(cls, path: Path) -> InitialTeamSelector:
        """Load shape-checked numeric data, without pickle or implicit actions."""
        if path.is_symlink() or not path.is_file():
            raise ValueError("ordinary selector artifact required")
        with np.load(path, allow_pickle=False) as data:
            if len(data.files) != 5 or set(data.files) != {
                "mean",
                "scale",
                "weight",
                "bias",
                "agent_ids",
            }:
                raise ValueError("exact selector artifact fields required")
            bias = data["bias"]
            ids = data["agent_ids"]
            if bias.shape != () or ids.shape != (_PLAYERS,) or ids.dtype.kind != "U":
                raise ValueError("scalar bias and canonical roster required")
            return cls(
                data["mean"],
                data["scale"],
                data["weight"],
                float(bias),
                tuple(str(value) for value in ids),
            )


def fit_initial_team_selector(
    features: NDArray[np.float64],
    labels: NDArray[np.int64],
    *,
    seed: int,
    agent_ids: tuple[str, ...],
    steps: int = 800,
    learning_rate: float = 0.05,
    l2: float = 0.01,
) -> InitialTeamSelector:
    """Fit on caller-provided train rows only; never split or relabel silently."""
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels)
    if (
        not _valid_ids(agent_ids)
        or x.ndim != 2
        or x.shape[1] != _FEATURES
        or len(x) < 4
        or y.shape != (len(x),)
        or not np.isfinite(x).all()
        or not np.issubdtype(y.dtype, np.integer)
        or np.count_nonzero(y == 0) < 2
        or np.count_nonzero(y == 1) < 2
        or np.any((y != 0) & (y != 1))
        or type(seed) is not int
        or not 0 <= seed < 2**32
        or type(steps) is not int
        or not 1 <= steps <= 100_000
        or not np.isfinite(learning_rate)
        or not 0 < learning_rate <= 1
        or not np.isfinite(l2)
        or not 0 <= l2 <= 1
    ):
        raise ValueError("finite binary training data with both classes required")
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < 1e-6] = 1.0
    normalized = (x - mean) / scale
    rng = np.random.Generator(np.random.PCG64(seed))
    weight = rng.normal(0.0, 0.01, _FEATURES)
    bias = 0.0
    target = y.astype(np.float64)
    for _ in range(steps):
        error = _probability(normalized @ weight + bias) - target
        weight -= learning_rate * (normalized.T @ error / len(x) + l2 * weight)
        bias -= learning_rate * float(error.mean())
    return InitialTeamSelector(mean, scale, weight, bias, agent_ids)
