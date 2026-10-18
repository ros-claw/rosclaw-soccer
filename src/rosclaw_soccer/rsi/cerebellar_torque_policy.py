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
TIME_AWARE_WIDTHS = (127, 64, 64, 29)
TIME_AWARE_SCHEMA = "soccer.rsi.cerebellar_torque_residual.v2"
TIME_AWARE_OBSERVATION = dict(
    schema="soccer.rsi.cerebellar_proprioception_and_clock.v1",
    v1_features_prefix=97,
    appended_joint_position_count=29,
    joint_position_scale_rad=2.0,
    appended_remaining_time_fraction=True,
    horizon_substeps=3000,
    substep_index_origin=0,
    input_is_before_integration=True,
    future_outcome_used=False,
)
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


def features(
    qpos: Any, qvel: Any, target: Any, previous_applied: Any, *, substep_index: int | None = None
) -> np.ndarray[Any, Any]:
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
    if substep_index is not None:
        if type(substep_index) is not int or not 0 <= substep_index < 3000:
            raise ValueError("before-integration substep index in [0, 3000) required")
        result = np.concatenate(
            (result, np.clip(q[7:36] / 2, -10, 10), [(3000 - substep_index) / 3000])
        )
    return result


def validate_model(model: dict[str, Any]) -> None:
    time_aware = type(model) is dict and model.get("schema") == TIME_AWARE_SCHEMA
    widths = TIME_AWARE_WIDTHS if time_aware else WIDTHS
    if (
        type(model) is not dict
        or model.get("schema") not in (SCHEMA, TIME_AWARE_SCHEMA)
        or model.get("architecture") != list(widths)
        or (time_aware and model.get("observation") != TIME_AWARE_OBSERVATION)
        or (not time_aware and "observation" in model)
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
    for layer, before, after in zip(model["layers"], widths[:-1], widths[1:], strict=True):
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


def initialize(*, seed: int, world_hash: str, time_aware: bool = False) -> dict[str, Any]:
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("explicit uint32 initialization seed required")
    if type(time_aware) is not bool:
        raise ValueError("explicit boolean model version choice required")
    widths = TIME_AWARE_WIDTHS if time_aware else WIDTHS
    rng = np.random.default_rng(seed)
    layers = []
    for index, (before, after) in enumerate(zip(widths[:-1], widths[1:], strict=True)):
        weight = (
            np.zeros((after, before))
            if index == 2
            else rng.normal(size=(after, before)) / np.sqrt(before)
        )
        layers.append(dict(weight=weight.tolist(), bias=np.zeros(after).tolist()))
    result = dict(
        schema=TIME_AWARE_SCHEMA if time_aware else SCHEMA,
        architecture=list(widths),
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
    if time_aware:
        result["observation"] = dict(TIME_AWARE_OBSERVATION)
    result["model_hash"] = hash_json(result)
    validate_model(result)
    return result


class CerebellarTorquePolicy:
    """Owned read-only weights, explicit applied history, optional episode RNG.

    A sampled episode must retain the same object across control frames. The
    declared RNG seed is not a learned parameter and is never reset implicitly.
    """

    def __init__(self, model: dict[str, Any], *, sampling_seed: int | None = None) -> None:
        validate_model(model)
        if sampling_seed is not None and (
            type(sampling_seed) is not int or not 0 <= sampling_seed < 2**32
        ):
            raise ValueError("optional explicit uint32 sampling seed required")
        self.model_hash = model["model_hash"]
        self.world_hash = model["world_hash"]
        self.optimizer_steps = model["optimizer_steps"]
        self.time_aware = model["schema"] == TIME_AWARE_SCHEMA
        self.sampling_seed = sampling_seed
        self.samples_drawn = 0
        self._rng = None if sampling_seed is None else np.random.default_rng(sampling_seed)
        self._sampling_transition: dict[str, Any] | None = None
        self._last_raw_mean: np.ndarray[Any, Any] | None = None
        self._layers = []
        for layer in model["layers"]:
            weight, bias = (
                np.array(layer[key], dtype=np.float64, copy=True) for key in ("weight", "bias")
            )
            weight.setflags(write=False)
            bias.setflags(write=False)
            self._layers.append((weight, bias))

    def offset(
        self,
        qpos: Any,
        qvel: Any,
        target: Any,
        previous_applied: Any,
        *,
        substep_index: int | None = None,
    ) -> np.ndarray[Any, Any]:
        if self.time_aware != (substep_index is not None):
            raise ValueError("explicit substep clock is required only for model v2")
        value = features(qpos, qvel, target, previous_applied, substep_index=substep_index)
        for index, (weight, bias) in enumerate(self._layers):
            value = weight @ value + bias
            if index < 2:
                value = np.tanh(value)
        latent = value
        self._last_raw_mean = value.copy()
        if self._rng is not None:
            noise = self._rng.normal(size=29)
            latent = value + 0.05 * noise
            standardized = (latent - value) / 0.05
            logp = float(np.sum(-0.5 * standardized**2 - np.log(0.05) - 0.5 * np.log(2 * np.pi)))
            self._sampling_transition = dict(
                mean=value.copy(),
                latent_action=latent.copy(),
                log_probability=logp,
                sample_index=self.samples_drawn,
            )
            self.samples_drawn += 1
        result: np.ndarray[Any, Any] = np.tanh(latent) * CAPS
        return result

    def last_raw_mean(self) -> np.ndarray[Any, Any]:
        if self._last_raw_mean is None:
            raise ValueError("neural torque policy has not been queried")
        return self._last_raw_mean.copy()

    def sampling_transition(self) -> dict[str, Any] | None:
        """Copy actual latent density record; it is not an actuator density."""
        if self._sampling_transition is None:
            return None
        return {
            key: value.copy() if isinstance(value, np.ndarray) else value
            for key, value in self._sampling_transition.items()
        }

    def sampling_contract(self) -> dict[str, Any]:
        return dict(
            schema="soccer.rsi.cerebellar_latent_sampling.v1",
            model_hash=self.model_hash,
            sampling_seed=self.sampling_seed,
            latent_std=0.05,
            generator="numpy_default_rng_PCG64",
            distribution="IID Gaussian latent before tanh and original hard torque projection",
            actuator_density_claimed=False,
            reset_per_control_frame=False,
            hardware_authorized=False,
            promotion_authorized=False,
        )
