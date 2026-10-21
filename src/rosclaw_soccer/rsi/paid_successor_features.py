"""CPU-causal student features; no source/case labels or teacher memory inputs."""

from typing import Any

import numpy as np
import torch


def successor_features(
    base: Any, raw: dict[str, np.ndarray], scale: torch.Tensor, *, teacher: bool
) -> tuple[np.ndarray, np.ndarray]:
    n = len(raw["NN_obs"])
    if (
        not 1 <= n <= 300
        or scale.shape != (29,)
        or scale.dtype != torch.float32
        or scale.device.type != "cpu"
        or not torch.isfinite(scale).all()
        or not (scale > 0).all()
    ):
        raise ValueError("bounded original29 scale and actual CPU trajectory required")
    state = (torch.zeros(2, 1, 128), torch.zeros(2, 1, 128))
    features, targets = [], []
    with torch.no_grad():
        for i, item in enumerate(raw["NN_obs"]):
            if item.shape != (160,) or item.dtype != np.float32 or not np.isfinite(item).all():
                raise ValueError("original160 measured observations required")
            if any(
                not np.array_equal(x.numpy(), raw[key][i])
                for x, key in zip(state, ("NN_h_before", "NN_c_before"), strict=True)
            ):
                raise ValueError("private own-memory causal chain differs")
            obs = torch.from_numpy(item.copy()).reshape(1, 1, 160)
            nominal, state, hidden = base.forward_with_features(obs, state)
            if not np.array_equal(nominal[0, 0].numpy(), raw["NN_nominal_action"][i]) or any(
                not np.array_equal(x.numpy(), raw[key][i])
                for x, key in zip(state, ("NN_h_after", "NN_c_after"), strict=True)
            ):
                raise ValueError("own base forward differs from physical history")
            actual = torch.from_numpy(raw["NN_action"][i].copy())
            target = (actual - nominal[0, 0]) * scale
            if not torch.isfinite(target).all() or target.abs().max() > 0.20001:
                raise ValueError("bounded actual29 joint-radian correction required")
            if not teacher and torch.count_nonzero(target):
                raise ValueError("protected successes must have genuine zero targets")
            f = torch.cat([(obs - base.mean) / base.denominator, hidden], dim=-1)[0, 0]
            if f.shape != (288,) or not torch.isfinite(f).all():
                raise ValueError("own normalized160 plus causal128 features required")
            features.append(f.numpy().copy())
            targets.append(target.numpy().copy())
    return np.stack(features), np.stack(targets)


def successor_student_config(seed: int) -> dict[str, Any]:
    if type(seed) is not int or seed not in (0, 1, 2):
        raise ValueError("one of three predeclared independent student seeds required")
    return dict(
        protocol="successor-contact-student-v1",
        seed_index=seed,
        seed=2026102200 + seed,
        epochs=256,
        steps_per_epoch=4,
        lr=0.0003,
        batch_rows=512,
        retention_weight=5.0,
        active_weight=5.0,
        gate_weight=0.001,
        gate_positive_weight=5.0,
        hard_negative_weight=0.005,
        hard_negative_margin=0.5,
        gate_retention_margin=0.1,
        gate_projection=True,
        max_grad_norm=1.0,
        bound_radians=0.2,
        features=288,
        original_observations=160,
        frozen_base=True,
        runtime_case_lookup=False,
        DEV_episodes_fit=0,
        memory_handoff="reset_for_kick_carry_own_base_for_stand",
    )


def staged_successor_config(seed: int) -> dict[str, Any]:
    config = successor_student_config(seed)
    config.update(
        protocol="successor-contact-student-v2",
        seed=2026102300 + seed,
        gate_projection=False,
        gate_training_steps=512,
        head_training_steps=1024,
        gate_positive_core_rad=0.005,
        gate_hard_negative_rows=256,
        gate_strategy="ranking_hard_negatives_then_CPU_bias_calibration_and_freeze",
        calibration_margin=0.1,
        gate_frozen_during_head_training=True,
    )
    return config
