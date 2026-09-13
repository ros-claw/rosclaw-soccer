"""Pure held-action PPO time-base conversion; no physics or policy authority.

Input episodes must be complete, causally terminated and zero padded. This is
not a bootstrap/truncated-rollout aggregator. Only decision-start likelihoods
are returned; intermediate likelihoods are diagnostics, not extra samples.
"""

import math
from collections.abc import Mapping

import numpy as np


def coarsen_held_action_rollout(
    data: Mapping[str, np.ndarray],
    *,
    period: int,
    gamma: float,
    observation_size: int,
    action_size: int,
    learning_feature: int,
) -> dict[str, np.ndarray]:
    """Sum micro rewards with gamma**j; caller uses gamma**period for PPO.

    Feature identity and policy/checkpoint compatibility remain caller-owned.
    The learning window and raw action must stay fixed across live substeps.
    Returned arrays are independent copies, never views into training evidence.
    """
    if (
        type(period) is not int
        or not 1 <= period <= 10
        or type(gamma) not in (int, float)
        or not math.isfinite(gamma)
        or not 0 < gamma <= 1
        or type(observation_size) is not int
        or not 1 <= observation_size <= 512
        or type(action_size) is not int
        or not 1 <= action_size <= 64
        or type(learning_feature) is not int
        or not 0 <= learning_feature < observation_size
    ):
        raise ValueError("explicit bounded time-base and feature contract required")
    required = {"obs", "raw", "logp", "value", "reward", "alive", "next_alive"}
    if (
        not isinstance(data, Mapping)
        or set(data) != required
        or not isinstance(data["reward"], np.ndarray)
        or data["reward"].ndim != 2
    ):
        raise ValueError("complete time/world rollout required")
    horizon, worlds = data["reward"].shape
    if (
        horizon % period
        or not period <= horizon <= 4095
        or not 1 <= worlds <= 4096
        or horizon * worlds > 65536
    ):
        raise ValueError("bounded complete held-action decisions required")
    for key, value in data.items():
        shape = (
            (horizon, worlds, observation_size if key == "obs" else action_size)
            if key in ("obs", "raw")
            else (horizon, worlds)
        )
        if (
            not isinstance(value, np.ndarray)
            or value.shape != shape
            or value.dtype != np.float32
            or not np.isfinite(value).all()
        ):
            raise ValueError("finite float32 rollout shapes required")
    if (
        any(not np.isin(data[key], [0.0, 1.0]).all() for key in ("alive", "next_alive"))
        or np.any(data["next_alive"] > data["alive"])
        or not np.array_equal(data["alive"][1:], data["next_alive"][:-1])
    ):
        raise ValueError("causal episode masks required")
    if np.any(data["next_alive"][-1] != 0) or any(
        np.any(value[data["alive"] == 0] != 0) for value in data.values()
    ):
        raise ValueError("terminal episodes and zero dead padding required")
    if not np.isin(data["obs"][..., learning_feature], [0.0, 1.0]).all():
        raise ValueError("binary learning-window feature required")
    for start in range(0, horizon, period):
        for offset in range(1, period):
            live = data["alive"][start + offset] > 0
            if not np.array_equal(
                data["raw"][start + offset, live], data["raw"][start, live]
            ) or not np.array_equal(
                data["obs"][start + offset, live, learning_feature],
                data["obs"][start, live, learning_feature],
            ):
                raise ValueError("action and learning window must be held within decisions")
    result = {key: data[key][::period].copy() for key in ("obs", "raw", "logp", "value", "alive")}
    result["next_alive"] = data["next_alive"][period - 1 :: period].copy()
    reward = np.zeros((horizon // period, worlds), dtype=np.float64)
    for offset in range(period):
        reward += (gamma**offset) * data["reward"][offset::period].astype(np.float64)
    if not np.isfinite(reward).all() or np.any(np.abs(reward) > np.finfo(np.float32).max):
        raise ValueError("aggregated rewards must remain representable as float32")
    result["reward"] = reward.astype(np.float32)
    return result
