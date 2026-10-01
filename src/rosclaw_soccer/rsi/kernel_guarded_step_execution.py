"""Causal compiled mean/exploration inference for kernel-protected motor heads."""

import copy
from pathlib import Path
from typing import Any, cast

import numpy as np

from rosclaw_soccer.rsi import kernel_guarded_step_network as network
from rosclaw_soccer.rsi.compiled_step_inference import CompiledStepMotor
from rosclaw_soccer.rsi.contact_motor_phase import make_policy
from rosclaw_soccer.rsi.contact_motor_primitive import CAP_RAD, SLEW_RAD
from rosclaw_soccer.rsi.step_motor_execution import make_preview as warm_preview
from rosclaw_soccer.rsi.step_motor_phase_context import ContactPhaseMemory
from rosclaw_soccer.rsi.stochastic_step_execution import features_at_frame
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json

SAMPLING_SCHEMA = "soccer.rsi.kernel_guarded_step_sampling.v1"


def make_sampling_view(model: dict[str, Any], *, seed: int, std: float) -> dict[str, Any]:
    network.validate_model(model)
    if (
        type(seed) is not int
        or not 0 <= seed < 2**32
        or type(std) not in (float, int)
        or not np.isfinite(std)
        or not 0.01 <= std <= 0.15
    ):
        raise ValueError("explicit bounded training-only Gaussian exploration required")
    view = dict(
        schema=SAMPLING_SCHEMA,
        mean_model=copy.deepcopy(model),
        seed=seed,
        std_raw=float(std),
        activation_ceiling="SIM_ONLY",
        training_only=True,
        source_hash=hash_bytes(Path(__file__).read_bytes()),
        **dict.fromkeys(network.FLAGS, False),
    )
    view["model_hash"] = hash_json(view)
    return view


def mean_model(model: dict[str, Any]) -> dict[str, Any]:
    if model.get("schema") == SAMPLING_SCHEMA:
        if (
            model.get("model_hash")
            != hash_json({k: v for k, v in model.items() if k != "model_hash"})
            or model.get("activation_ceiling") != "SIM_ONLY"
            or model.get("training_only") is not True
            or model.get("source_hash") != hash_bytes(Path(__file__).read_bytes())
            or any(model.get(k) is not False for k in network.FLAGS)
            or type(model.get("seed")) is not int
            or not 0 <= model["seed"] < 2**32
            or type(model.get("std_raw")) not in (float, int)
            or not np.isfinite(model["std_raw"])
            or not 0.01 <= model["std_raw"] <= 0.15
        ):
            raise ValueError("unsealed kernel motor exploration view")
        current = model["mean_model"]
    else:
        current = model
    network.validate_model(current)
    return cast(dict[str, Any], current)


def make_preview(model: dict[str, Any]) -> dict[str, Any]:
    current = mean_model(model)
    policy = make_policy(np.zeros((3, 12)), 0.25, model["model_hash"])
    policy["execution_profile"] = "causal_per_frame_neural_residual"
    policy["step_motor_proof"] = dict(
        schema="soccer.rsi.kernel_step_preview.v1",
        model=model,
        decision_start_frame=30,
        nominal_target_is_pre_motor=True,
        force_input="previous_completed_frame",
        execution_source_hash=hash_bytes(Path(__file__).read_bytes()),
        qualification="UNQUALIFIED_SIM_TRAINING"
        if model.get("training_only")
        else "UNQUALIFIED_SIM_COUNTERFACTUAL",
        promotion_authorized=False,
    )
    policy["kernel_motor_proof"] = dict(
        guard_hash=current["anchor_guard"]["guard_hash"],
        mean_model_hash=current["model_hash"],
        promotion_authorized=False,
    )
    policy.pop("policy_hash")
    policy["policy_hash"] = hash_json(policy)
    return policy


class CompiledKernelStepMotor:
    def __init__(self, policy: dict[str, Any]) -> None:
        wrapped = copy.deepcopy(policy["step_motor_proof"]["model"])
        if make_preview(wrapped) != policy:
            raise ValueError("kernel motor preview lost sealed execution binding")
        model = mean_model(wrapped)
        self._guard = network.validate_model(model)
        base = model["encoder"]["base_model"]
        self._warm = CompiledStepMotor.from_legacy_preview(warm_preview(base))
        self._mean = np.asarray(base["mean"], dtype=np.float64)
        self._scale = np.asarray(base["scale"], dtype=np.float64)
        self._layers = [
            (np.asarray(v["weight"]), np.asarray(v["bias"])) for v in base["actor"]["layers"][:-1]
        ]
        random = model["encoder"]["frozen_random_features"]
        self._random = (np.asarray(random["weight"]), np.asarray(random["bias"]))
        self._head = np.asarray(model["actor_readout"], dtype=np.float64)
        for value in (
            self._mean,
            self._scale,
            self._head,
            *self._random,
            *(a for pair in self._layers for a in pair),
        ):
            value.flags.writeable = False
        self._sampling = wrapped if wrapped["schema"] == SAMPLING_SCHEMA else None
        self._memory = ContactPhaseMemory()
        self._policy_hash = policy["policy_hash"]

    def features(self, observation: Any) -> np.ndarray[Any, Any]:
        x = np.asarray(observation, dtype=np.float64)
        if x.shape != (134,) or not np.isfinite(x).all():
            raise ValueError("finite causal motor observation required")
        norm = np.clip((x - self._mean) / self._scale, -8, 8)
        hidden = norm
        for weight, bias in self._layers:
            hidden = np.tanh(weight @ hidden + bias)
        return np.concatenate(
            (norm, hidden, np.tanh(self._random[0] @ norm + self._random[1]), [1.0])
        )

    def raw_mean(self, observation: Any, phase: int) -> np.ndarray[Any, Any]:
        if type(phase) is not int or not 0 <= phase <= 2:
            raise ValueError("causal physical contact phase required")
        phi = self.features(observation)
        mean = self._warm.raw_mean(observation) + 0.05 * self._guard.gate(phi[:134]) * np.tanh(
            self._head[phase] @ phi
        )
        if not np.isfinite(mean).all():
            raise ValueError("nonfinite kernel motor mean")
        return np.asarray(mean, dtype=np.float64)

    def latent_sample(self, observation: Any, frame: int, phase: int) -> tuple[Any, float]:
        if self._sampling is None or type(frame) is not int or not 30 <= frame < 3000:
            raise ValueError("sealed training exploration after decision boundary required")
        mean = self.raw_mean(observation, phase)
        view = self._sampling
        rng = np.random.default_rng(np.random.SeedSequence(view["seed"], spawn_key=(frame,)))
        noise = rng.normal(size=12)
        std = view["std_raw"]
        return mean + std * noise, float(
            np.sum(-0.5 * noise**2 - np.log(std) - 0.5 * np.log(2 * np.pi))
        )

    def delta_at_frame(
        self,
        policy: dict[str, Any],
        body: Any,
        *,
        frame: int,
        nominal_target: Any,
        baseline: Any,
        limits: Any,
        previous: Any,
        previous_contact_forces: Any,
    ) -> np.ndarray[Any, Any]:
        if policy.get("policy_hash") != self._policy_hash:
            raise ValueError("running kernel motor cannot switch commitment")
        phase = self._memory.advance(frame, previous_contact_forces)
        x = features_at_frame(
            body,
            frame=frame,
            nominal_target=nominal_target,
            previous=previous,
            previous_contact_forces=previous_contact_forces,
        )
        base, bounds, prior = [
            np.asarray(v, dtype=np.float64) for v in (baseline, limits, previous)
        ]
        if (
            base.shape != (12,)
            or bounds.shape != (12, 2)
            or prior.shape != (12,)
            or not all(np.isfinite(v).all() for v in (base, bounds, prior))
            or np.any(bounds[:, 0] >= bounds[:, 1])
            or np.max(np.abs(prior)) > CAP_RAD + 1e-5
        ):
            raise ValueError("finite measured motor envelope required")
        if frame < 30:
            if np.any(prior != 0):
                raise ValueError("kernel motor cannot predate decision boundary")
            return np.zeros(12)
        mean = self.latent_sample(x, frame, phase)[0] if self._sampling else self.raw_mean(x, phase)
        desired = CAP_RAD * np.tanh(mean)
        proposed = prior + np.clip(desired - prior, -SLEW_RAD, SLEW_RAD)
        lower = np.maximum(np.minimum(0.0, bounds[:, 0] - base), -CAP_RAD)
        upper = np.minimum(np.maximum(0.0, bounds[:, 1] - base), CAP_RAD)
        return np.asarray(np.clip(proposed, lower, upper), dtype=np.float64)
