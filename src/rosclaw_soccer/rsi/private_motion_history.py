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
    option_keys = {
        "NN_option_before",
        "NN_option_after",
        "NN_option_controls_before",
        "NN_option_controls_after",
        "NN_option_generation",
    }
    if any(option_keys & raw.keys() for raw in (kick, stand)):
        if any(not option_keys <= raw.keys() for raw in (kick, stand)):
            raise ValueError(
                "complete explicit option history required in BOTH motion and recovery"
            )
        for raw, length in zip((kick, stand), lengths, strict=True):
            for key in option_keys:
                shape, dtype = (
                    ((length, 32), np.uint8)
                    if key == "NN_option_generation"
                    else ((length,), np.int64)
                )
                if raw[key].shape != shape or raw[key].dtype != dtype:
                    raise ValueError("exact numeric private option trace contract required")
        option = {k: np.concatenate([kick[k], stand[k]], axis=0) for k in option_keys}
        total = sum(lengths)
        before, after = option["NN_option_before"], option["NN_option_after"]
        gen = option["NN_option_generation"]
        if (
            total > 512
            or before[0] != -1
            or (after < 0).any()
            or (after > 3).any()
            or not np.array_equal(before[1:], after[:-1])
            or not (after == after[0]).all()
            or not np.array_equal(option["NN_option_controls_before"], np.arange(total))
            or not np.array_equal(option["NN_option_controls_after"], np.arange(1, total + 1))
            or not (gen == gen[0]).all()
        ):
            raise ValueError(
                "fresh-to-recovery option commitment, age and generation continuity required"
            )
        result.update(option)
    return result
