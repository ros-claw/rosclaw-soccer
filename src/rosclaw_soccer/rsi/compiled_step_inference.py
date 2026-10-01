"""Compile a sealed SIM-only network once; infer without per-frame JSON hashing.

This changes representation, not weights, observations, action bounds or noise.
Numeric arrays are private copies. Independent reviewers compile their own
instance from the sealed report, then reconstruct every actual motor action.
"""

import copy
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.rsi.contact_motor_phase import make_policy
from rosclaw_soccer.rsi.contact_motor_primitive import CAP_RAD, SLEW_RAD
from rosclaw_soccer.rsi.stochastic_step_execution import features_at_frame
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json

SCHEMA = "soccer.rsi.compiled_step_motor_decoder.v1"


def _base_preview(base: dict[str, Any]) -> dict[str, Any]:
    if base.get("schema") == "soccer.rsi.online_step_motor_mc_ppo.v1":
        from rosclaw_soccer.rsi.online_step_execution import make_preview as preview
    elif "training_sampling" in base:
        from rosclaw_soccer.rsi.stochastic_step_execution import make_preview as preview
    else:
        from rosclaw_soccer.rsi.step_motor_execution import make_preview as preview
    return preview(base)


def make_model(base: dict[str, Any]) -> dict[str, Any]:
    _base_preview(base)
    model = dict(
        schema=SCHEMA,
        base_model=copy.deepcopy(base),
        activation_ceiling="SIM_ONLY",
        learning_kind="compilation_only_no_weight_update",
        source_hash=hash_bytes(Path(__file__).read_bytes()),
        promotion_authorized=False,
        runtime_execution_authorized=False,
        hardware_authorized=False,
        fresh_holdout_open_authorized=False,
    )
    model["model_hash"] = hash_json(model)
    return model


def make_preview(model: dict[str, Any]) -> dict[str, Any]:
    if (
        model.get("schema") != SCHEMA
        or model.get("activation_ceiling") != "SIM_ONLY"
        or model.get("learning_kind") != "compilation_only_no_weight_update"
        or model.get("source_hash") != hash_bytes(Path(__file__).read_bytes())
        or any(
            model.get(k) is not False
            for k in (
                "promotion_authorized",
                "runtime_execution_authorized",
                "hardware_authorized",
                "fresh_holdout_open_authorized",
            )
        )
        or model.get("model_hash")
        != hash_json({k: v for k, v in model.items() if k != "model_hash"})
    ):
        raise ValueError("sealed SIM-only compiled decoder required")
    legacy = _base_preview(model["base_model"])
    policy = make_policy(np.zeros((3, 12)), 0.25, model["model_hash"])
    policy["execution_profile"] = "causal_per_frame_neural_residual"
    policy["step_motor_proof"] = dict(
        schema="soccer.rsi.compiled_step_motor_sim_preview.v1",
        model=model,
        decision_start_frame=30,
        nominal_target_is_pre_motor=True,
        force_input="previous_completed_frame",
        qualification="UNQUALIFIED_SIM_COUNTERFACTUAL",
        execution_source_hash=hash_bytes(Path(__file__).read_bytes()),
        promotion_authorized=False,
    )
    policy["compiled_motor_proof"] = dict(
        base_policy_hash=legacy["policy_hash"],
        base_model_hash=model["base_model"]["model_hash"],
        compilation_changes_weights=False,
        promotion_authorized=False,
    )
    policy.pop("policy_hash")
    policy["policy_hash"] = hash_json(policy)
    return policy


class CompiledStepMotor:
    """Locally immutable numeric decoder, constructed before a closed-loop run."""

    def __init__(self, policy: dict[str, Any]) -> None:
        model = copy.deepcopy(policy["step_motor_proof"]["model"])
        if make_preview(model) != policy:
            raise ValueError("compiled preview differs from sealed network")
        self._policy_hash = policy["policy_hash"]
        base = model["base_model"]
        self.sampling = copy.deepcopy(base.get("training_sampling"))
        if base.get("schema") == "soccer.rsi.online_step_motor_mc_ppo.v1":
            from rosclaw_soccer.rsi.step_motor_ppo import numeric_view

            base = numeric_view(base)
        self.mean = np.asarray(base["mean"], dtype=np.float64).copy()
        self.scale = np.asarray(base["scale"], dtype=np.float64).copy()
        self.layers = [
            (
                np.asarray(layer["weight"], dtype=np.float64).copy(),
                np.asarray(layer["bias"], dtype=np.float64).copy(),
            )
            for layer in base["actor"]["layers"]
        ]
        for array in (self.mean, self.scale, *(a for layer in self.layers for a in layer)):
            array.flags.writeable = False

    @classmethod
    def from_legacy_preview(cls, policy: dict[str, Any]) -> "CompiledStepMotor":
        """Equivalent audit decoder; never claim the old run used compilation."""
        base = policy["step_motor_proof"]["model"]
        if _base_preview(base) != policy:
            raise ValueError("legacy preview differs from sealed original execution law")
        compiled = cls(make_preview(make_model(base)))
        compiled._policy_hash = policy["policy_hash"]
        return compiled

    def raw_mean(self, features: Any) -> np.ndarray[Any, Any]:
        observation = np.asarray(features, dtype=np.float64)
        if observation.shape != (134,) or not np.isfinite(observation).all():
            raise ValueError("finite causal compiled features required")
        value = np.clip((observation - self.mean) / self.scale, -8, 8)
        for index, (weight, bias) in enumerate(self.layers):
            value = weight @ value + bias
            if index < len(self.layers) - 1:
                value = np.tanh(value)
        if not np.isfinite(value).all():
            raise ValueError("nonfinite compiled neural mean")
        return np.asarray(value, dtype=np.float64)

    def latent_sample(self, features: Any, frame: int) -> tuple[np.ndarray[Any, Any], float]:
        if self.sampling is None or type(frame) is not int or not 30 <= frame < 3000:
            raise ValueError("compiled latent sample requires a causal sealed exploration law")
        mean = self.raw_mean(features)
        rng = np.random.default_rng(
            np.random.SeedSequence(self.sampling["seed"], spawn_key=(frame,))
        )
        noise = rng.normal(size=12)
        std = self.sampling["std_raw"]
        density = float(np.sum(-0.5 * noise**2 - np.log(std) - 0.5 * np.log(2 * np.pi)))
        return mean + std * noise, density

    def delta_at_frame(
        self, policy: dict[str, Any], body: Any, **boundary: Any
    ) -> np.ndarray[Any, Any]:
        if policy.get("policy_hash") != self._policy_hash:
            raise ValueError("compiled decoder cannot switch policy commitment")
        return self.predict(body, **boundary)

    def predict(
        self,
        body: Any,
        *,
        frame: int,
        nominal_target: Any,
        baseline: Any,
        limits: Any,
        previous: Any,
        previous_contact_forces: Any,
    ) -> np.ndarray[Any, Any]:
        if type(frame) is not int or not 0 <= frame < 3000:
            raise ValueError("finite causal frame boundary required")
        observation = features_at_frame(
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
            raise ValueError("bounded measured compiled motor boundary required")
        if frame < 30:
            if np.any(prior != 0):
                raise ValueError("motor cannot precede causal decision boundary")
            return np.zeros(12)
        value = (
            self.raw_mean(observation)
            if self.sampling is None
            else self.latent_sample(observation, frame)[0]
        )
        desired = CAP_RAD * np.tanh(value)
        proposed = prior + np.clip(desired - prior, -SLEW_RAD, SLEW_RAD)
        lower = np.maximum(np.minimum(0.0, bounds[:, 0] - base), -CAP_RAD)
        upper = np.minimum(np.maximum(0.0, bounds[:, 1] - base), CAP_RAD)
        return np.asarray(np.clip(proposed, lower, upper), dtype=np.float64)
