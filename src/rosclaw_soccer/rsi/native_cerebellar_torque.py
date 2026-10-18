"""Explicit SIM-only neural torque layer in the adaptive native control loop.

No robot transport or generic callback. Each episode owns its compiled network,
RNG and applied history; the Foundation and original motor policy still update
from that episode's current body at their original control rate.
"""

import copy
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.rsi import cerebellar_torque_policy as policy_module
from rosclaw_soccer.rsi.cerebellar_torque_policy import CerebellarTorquePolicy, validate_model
from rosclaw_soccer.sim.contracts import G1_HARD_TORQUE_LIMITS, hash_bytes, hash_json

SCHEMA = "soccer.rsi.native_adaptive_cerebellar_torque.v1"
WIDTHS = {
    "cerebellar_canonical_qpos_before_substep": 43,
    "cerebellar_canonical_qvel_before_substep": 41,
    "cerebellar_input_features": 97,
    "cerebellar_sampling_mean": 29,
    "cerebellar_sampling_latent_action": 29,
    "cerebellar_sampling_log_probability": 1,
    "cerebellar_sampling_sample_index": 1,
    "cerebellar_sampling_performed": 1,
    "requested_cerebellar_torque_offset_nm": 29,
    "applied_cerebellar_torque_offset_nm": 29,
}


def validate_native_request(
    model: dict[str, Any] | None,
    sampling_seed: int | None,
    *,
    partition: str,
    captured: bool,
    foundation_only: bool,
    recurrent_factory: Any,
    mixed_probe: bool,
) -> None:
    if model is None:
        if sampling_seed is not None:
            raise ValueError("cerebellar sampling requires an explicit model")
        return
    validate_model(model)
    if sampling_seed is not None and (
        type(sampling_seed) is not int or not 0 <= sampling_seed < 2**32
    ):
        raise ValueError("explicit uint32 cerebellar sampling seed required")
    if (
        partition != "DECLARED_DEVELOPMENT_DIAGNOSTIC"
        or captured is not True
        or foundation_only is not False
        or recurrent_factory is None
        or mixed_probe
    ):
        raise ValueError("captured unmixed adaptive SIM development episode required")
    from rosclaw_soccer.rsi.recurrent_sampling_episode_factory import (
        RecurrentSamplingEpisodeFactory,
    )

    if type(recurrent_factory) is not RecurrentSamplingEpisodeFactory:
        raise ValueError("exact original recurrent sampling factory required")
    try:
        recurrent_factory._stable()
    except AttributeError as error:
        raise ValueError("allocated stable original recurrent factory required") from error


class NativeCerebellarTorque:
    def __init__(
        self, model: dict[str, Any], *, world_hash: str, sampling_seed: int | None
    ) -> None:
        validate_model(model)
        if world_hash != model["world_hash"]:
            raise ValueError("neural torque layer must bind the actual compiled world")
        self._model = copy.deepcopy(model)
        self._policy = CerebellarTorquePolicy(self._model, sampling_seed=sampling_seed)
        self._previous = np.zeros(29)
        self.calls = 0

    def contract(self) -> dict[str, Any]:
        value = dict(
            schema=SCHEMA,
            model=copy.deepcopy(self._model),
            model_hash=self._policy.model_hash,
            world_hash=self._policy.world_hash,
            sampling=self._policy.sampling_contract(),
            source_hashes=dict(
                bridge=hash_bytes(Path(__file__).read_bytes()),
                policy=hash_bytes(Path(policy_module.__file__).read_bytes()),
            ),
            foundation_control_dt_s=0.02,
            torque_control_dt_s=0.002,
            foundation_and_parent_motor_use_current_body=True,
            episode_applied_history_initial_zero=True,
            deterministic_logp_zero_is_placeholder_not_density=True,
            activation_ceiling="SIM_ONLY",
            runtime_execution_authorized=False,
            teacher_qualified=False,
            hardware_authorized=False,
            promotion_authorized=False,
        )
        return dict(value, contract_hash=hash_json(value))

    def advance(
        self, qpos: np.ndarray, qvel: np.ndarray, target: np.ndarray, pd_requested: np.ndarray
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        raw = np.asarray(pd_requested)
        if raw.shape != (29,) or raw.dtype != np.float64 or not np.isfinite(raw).all():
            raise ValueError("finite canonical preclip PD request required")
        feature = policy_module.features(qpos, qvel, target, self._previous)
        offset = self._policy.offset(qpos, qvel, target, self._previous)
        mean = self._policy.last_raw_mean()
        sample = self._policy.sampling_transition()
        limits = np.asarray(G1_HARD_TORQUE_LIMITS)
        torque = np.clip(raw + offset, -limits, limits)
        applied = torque - np.clip(raw, -limits, limits)
        record = {
            "cerebellar_canonical_qpos_before_substep": qpos.copy(),
            "cerebellar_canonical_qvel_before_substep": qvel.copy(),
            "cerebellar_input_features": feature,
            "cerebellar_sampling_mean": mean,
            "cerebellar_sampling_latent_action": mean
            if sample is None
            else sample["latent_action"],
            "cerebellar_sampling_log_probability": np.asarray(
                [0.0 if sample is None else sample["log_probability"]], dtype=np.float64
            ),
            "cerebellar_sampling_sample_index": np.asarray(
                [-1 if sample is None else sample["sample_index"]], dtype=np.int64
            ),
            "cerebellar_sampling_performed": np.asarray([sample is not None], dtype=np.bool_),
            "requested_cerebellar_torque_offset_nm": offset.copy(),
            "applied_cerebellar_torque_offset_nm": applied.copy(),
        }
        self._previous = applied.copy()
        self.calls += 1
        return torque, record
