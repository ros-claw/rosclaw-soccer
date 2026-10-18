"""SIM research 500 Hz neural torque residual; not a qualified motor skill.

Foundation/PD remain the predecessor. Zero-initialized output preserves it.
Inputs are current proprioception, target error and previous applied residual,
not ball truth, ground force, solver constraints or future state. Authentication
of datasets, optimization and full closed-loop qualification are caller duties.
"""

import re
from typing import Any

import numpy as np

from rosclaw_soccer.sim.contracts import G1_HARD_TORQUE_LIMITS, hash_json

WIDTHS = (97, 64, 64, 29)
SCHEMA = "soccer.rsi.cerebellar_torque_residual.v1"
CAPS = np.minimum(2.0, 0.04 * np.asarray(G1_HARD_TORQUE_LIMITS))
CAPS.setflags(write=False)


def _array(value: Any, shape: tuple[int, ...]) -> np.ndarray[Any, Any]:
    raw = np.asarray(value)
    if (
        raw.shape != shape
        or raw.dtype != np.float64
        or not np.isfinite(raw).all()
        or np.max(np.abs(raw)) > 1e6
    ):
        raise ValueError("complete finite float64 canonical proprioceptive arrays required")
    return raw


def features(qpos: Any, qvel: Any, target: Any, previous_applied: Any) -> np.ndarray[Any, Any]:
    q, v, t, previous = (
        _array(value, shape)
        for value, shape in (
            (qpos, (43,)),
            (qvel, (41,)),
            (target, (29,)),
            (previous_applied, (29,)),
        )
    )
    if not np.isclose(np.linalg.norm(q[3:7]), 1, rtol=0, atol=1e-6):
        raise ValueError("unit current root quaternion required")
    if np.any(np.abs(previous) > CAPS + 1e-12):
        raise ValueError("previous actually applied torque residual exceeds fixed cap")
    quaternion = q[3:7].copy()
    # Quaternion double cover: a fixed first-nonzero sign, no state mutation.
    nonzero = np.flatnonzero(quaternion != 0)
    if quaternion[nonzero[0]] < 0:
        quaternion *= -1
    value = np.concatenate(
        (
            (t - q[7:36]) / 0.5,
            v[6:35] / 10,
            quaternion,
            v[:6] / np.asarray((2, 2, 2, 3, 3, 3)),
            previous / CAPS,
        )
    )
    result: np.ndarray[Any, Any] = np.clip(value, -10, 10)
    return result


def validate_model(model: dict[str, Any]) -> None:
    if (
        type(model) is not dict
        or model.get("schema") != SCHEMA
        or model.get("architecture") != list(WIDTHS)
        or model.get("activation_ceiling") != "SIM_ONLY"
        or model.get("control_dt_s") != 0.002
        or type(model.get("optimizer_steps")) is not int
        or not 0 <= model["optimizer_steps"] <= 100000
        or type(model.get("seed")) is not int
        or not 0 <= model["seed"] < 2**32
        or type(model.get("world_hash")) is not str
        or re.fullmatch(r"sha256:[0-9a-f]{64}", model["world_hash"]) is None
        or model.get("offset_caps_nm") != CAPS.tolist()
        or any(
            model.get(key) is not False
            for key in (
                "runtime_execution_authorized",
                "teacher_qualified",
                "promotion_authorized",
                "hardware_authorized",
                "fresh_holdout_open_authorized",
            )
        )
        or type(model.get("layers")) is not list
        or len(model["layers"]) != 3
    ):
        raise ValueError("bounded SIM neural torque residual model required")
    for layer, before, after in zip(model["layers"], WIDTHS[:-1], WIDTHS[1:], strict=True):
        if type(layer) is not dict or set(layer) != {"weight", "bias"}:
            raise ValueError("complete fixed neural torque layer required")
        for key, shape in (("weight", (after, before)), ("bias", (after,))):
            value = np.asarray(layer[key])
            if (
                value.shape != shape
                or value.dtype.kind not in "fi"
                or not np.isfinite(value).all()
                or np.max(np.abs(value)) > 100
            ):
                raise ValueError("finite bounded fixed-shape neural torque parameters required")
    if model.get("model_hash") != hash_json({k: v for k, v in model.items() if k != "model_hash"}):
        raise ValueError("whole neural torque model seal differs")


def initialize(*, seed: int, world_hash: str) -> dict[str, Any]:
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("explicit uint32 initialization seed required")
    rng = np.random.default_rng(seed)
    layers = []
    for index, (before, after) in enumerate(zip(WIDTHS[:-1], WIDTHS[1:], strict=True)):
        weight = (
            np.zeros((after, before))
            if index == 2
            else rng.normal(size=(after, before)) / np.sqrt(before)
        )
        layers.append(dict(weight=weight.tolist(), bias=np.zeros(after).tolist()))
    result = dict(
        schema=SCHEMA,
        architecture=list(WIDTHS),
        seed=seed,
        world_hash=world_hash,
        activation_ceiling="SIM_ONLY",
        control_dt_s=0.002,
        offset_caps_nm=CAPS.tolist(),
        optimizer_steps=0,
        layers=layers,
        runtime_execution_authorized=False,
        teacher_qualified=False,
        promotion_authorized=False,
        hardware_authorized=False,
        fresh_holdout_open_authorized=False,
    )
    result["model_hash"] = hash_json(result)
    validate_model(result)
    return result


class CerebellarTorquePolicy:
    """Compile owned, read-only weights. Per-episode history stays explicit."""

    def __init__(self, model: dict[str, Any]) -> None:
        validate_model(model)
        self.model_hash = model["model_hash"]
        self.world_hash = model["world_hash"]
        self.optimizer_steps = model["optimizer_steps"]
        self._layers = []
        for layer in model["layers"]:
            weight, bias = (
                np.array(layer[key], dtype=np.float64, copy=True) for key in ("weight", "bias")
            )
            weight.setflags(write=False)
            bias.setflags(write=False)
            self._layers.append((weight, bias))

    def offset(
        self, qpos: Any, qvel: Any, target: Any, previous_applied: Any
    ) -> np.ndarray[Any, Any]:
        value = features(qpos, qvel, target, previous_applied)
        for weight, bias in self._layers:
            value = np.tanh(weight @ value + bias)
        return value * CAPS
