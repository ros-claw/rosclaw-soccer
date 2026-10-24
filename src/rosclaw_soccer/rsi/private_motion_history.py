"""Whole recorded private histories are witnesses, never recurrent inputs."""

from typing import Any

import numpy as np


def join_private_motion_history(
    kick: dict[str, Any], stand: dict[str, Any]
) -> dict[str, np.ndarray]:
    shapes = dict(
        NN_obs=(160,),
        NN_action=(29,),
        NN_h_before=(2, 1, 256),
        NN_c_before=(2, 1, 256),
        NN_h_after=(2, 1, 256),
        NN_c_after=(2, 1, 256),
    )
    lengths = [len(raw["NN_obs"]) for raw in (kick, stand)]
    if not 1 <= lengths[0] <= 512 or lengths[1] != 100:
        raise ValueError("complete bounded motion and100 real neural recovery rows required")
    for raw, length in zip((kick, stand), lengths, strict=True):
        for key, shape in shapes.items():
            x = raw[key]
            if (
                not isinstance(x, np.ndarray)
                or x.shape != (length, *shape)
                or x.dtype != np.float32
                or not np.isfinite(x).all()
            ):
                raise ValueError("finite whole256 private-state trace required")
    result = {k: np.concatenate([kick[k], stand[k]], axis=0) for k in shapes}
    for prefix in ("h", "c"):
        before, after = result[f"NN_{prefix}_before"], result[f"NN_{prefix}_after"]
        if before[0].any() or not np.array_equal(before[1:], after[:-1]):
            raise ValueError("private zero-to-whole-motion-to-recovery continuity required")
    return result
