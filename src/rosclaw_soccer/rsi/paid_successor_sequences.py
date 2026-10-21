"""Offline paired-world recurrent supervision; no physics or runtime routing."""

from typing import Any

import numpy as np
import torch


def paired_arrays(first: Any, second: Any) -> dict[str, np.ndarray]:
    """Keep both actual skills and their boundary, without fabricated gap frames."""
    for z in (first, second):
        n = len(z["NN_obs"])
        if (
            not 1 <= n <= 300
            or z["NN_obs"].shape != (n, 160)
            or z["NN_action"].shape != (n, 29)
            or any(
                z[k].dtype != np.float32 or not np.isfinite(z[k]).all()
                for k in ("NN_obs", "NN_action")
            )
        ):
            raise ValueError("complete finite float32 unpadded physical skill rows required")
        if "PLAN_pose_offset" in z and (
            z["PLAN_pose_offset"].shape != (n, 6) or not np.isfinite(z["PLAN_pose_offset"]).all()
        ):
            raise ValueError("optional teacher contact mask must match actual neural rows")
    if (
        not np.array_equal(first["NN_action"][-1], second["NN_obs"][0, 125:154])
        or float(first["initial_time"]) != 0
        or float(second["initial_time"]) <= float(first["time_before"][-1])
    ):
        raise ValueError("same complete world's actual action and clock handoff required")
    return dict(
        obs=np.concatenate([first["NN_obs"], second["NN_obs"]]),
        action=np.concatenate([first["NN_action"], second["NN_action"]]),
        active=np.concatenate(
            [
                np.any(z["PLAN_pose_offset"] != 0, axis=1)
                if "PLAN_pose_offset" in z
                else np.zeros(len(z["NN_obs"]), dtype=bool)
                for z in (first, second)
            ]
        ),
        boundary=np.asarray(len(first["NN_obs"]), dtype=np.int64),
    )


def paired_forward(model, obs, boundary: int, *, carry: bool):
    """Private h/c is continuous only in the explicitly trained carry protocol.

    Held reference tails and the ball drop have no actor observations in these
    records. Memory stays constant across that measured observation gap; no fake
    zero observations, additional recurrent updates, or hidden teacher states.
    """
    if not 0 < boundary < len(obs) or obs.ndim != 2 or obs.shape[1] != 160:
        raise ValueError("explicit nonempty paired skill boundary required")
    zero = (
        torch.zeros(2, 1, 128, device=obs.device),
        torch.zeros(2, 1, 128, device=obs.device),
    )
    if carry:
        prediction, state = model(obs[:, None], zero)
        return prediction[:, 0], state
    first, _ = model(obs[:boundary, None], zero)
    second, state = model(obs[boundary:, None], zero)
    return torch.cat([first[:, 0], second[:, 0]]), state
