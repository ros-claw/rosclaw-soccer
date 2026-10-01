"""Bounded SIM-only Gaussian latent-action exploration for a 50 Hz actor.

Normal samples precede tanh and deterministic joint/slew shielding. Likelihoods
refer to latent samples, not the non-invertible shielded motor output. No
future physics is read; random seeds select exploration, never actor features.
"""

from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.rsi import step_motor_network as network
from rosclaw_soccer.rsi.contact_motor_primitive import CAP_RAD, SLEW_RAD
from rosclaw_soccer.rsi.step_motor_execution import make_preview as deterministic_preview
from rosclaw_soccer.rsi.step_motor_features import feature_vector
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json


def make_sampling_view(base: dict[str, Any], *, seed: int, std: float) -> dict[str, Any]:
    network.validate_model(base)
    if (
        type(seed) is not int
        or not 0 <= seed < 2**32
        or type(std) not in (float, int)
        or not np.isfinite(std)
        or not 0.01 <= std <= 0.15
        or "training_sampling" in base
    ):
        raise ValueError("bounded explicit training exploration required")
    view = {k: v for k, v in base.items() if k != "model_hash"}
    view["training_sampling"] = dict(
        seed=seed,
        std_raw=float(std),
        base_model_hash=base["model_hash"],
        source_hash=hash_bytes(Path(__file__).read_bytes()),
        training_only=True,
    )
    view["model_hash"] = hash_json(view)
    network.validate_model(view)
    return view


def make_preview(model: dict[str, Any]) -> dict[str, Any]:
    network.validate_model(model)
    sampling = model.get("training_sampling", {})
    if (
        sampling.get("training_only") is not True
        or sampling.get("source_hash") != hash_bytes(Path(__file__).read_bytes())
        or type(sampling.get("seed")) is not int
        or not 0 <= sampling["seed"] < 2**32
        or type(sampling.get("std_raw")) not in (float, int)
        or not 0.01 <= sampling["std_raw"] <= 0.15
    ):
        raise ValueError("unsealed stochastic execution contract")
    base = {k: v for k, v in model.items() if k not in ("model_hash", "training_sampling")}
    base["model_hash"] = hash_json(base)
    if base["model_hash"] != sampling["base_model_hash"]:
        raise ValueError("exploration view changed mean policy weights")
    policy = deterministic_preview(model)
    policy["stochastic_motor_proof"] = dict(
        schema="soccer.rsi.stochastic_step_preview.v1",
        sampling=sampling,
        likelihood="unshielded Gaussian before tanh",
        qualification="UNQUALIFIED_SIM_TRAINING",
        promotion_authorized=False,
    )
    policy.pop("policy_hash")
    policy["policy_hash"] = hash_json(policy)
    return policy


def features_at_frame(
    body: Any, *, frame: int, nominal_target: Any, previous: Any, previous_contact_forces: Any
) -> np.ndarray[Any, Any]:
    return feature_vector(
        joint_position=np.asarray(body["joint_position_rad"])[frame, 0],
        joint_velocity=np.asarray(body["joint_velocity_rad_s"])[frame, 0],
        root_pose_xyzw=np.asarray(body["root_pose_xyzw_m"])[frame, 0],
        root_velocity_world=np.asarray(body["root_velocity_world"])[frame, 0],
        ball_position_world=np.asarray(body["ball_position_before_step_m"])[frame, 0],
        ball_velocity_world=np.asarray(body["ball_linear_velocity_before_step_m_s"])[frame, 0],
        geometry_position_world=np.asarray(body["foot_geometry_position_before_step_m"])[frame, 0],
        nominal_target=nominal_target,
        previous_motor_delta=previous,
        previous_contact_forces=previous_contact_forces,
        frame=frame,
    )


def raw_actor_mean(model: dict[str, Any], features: Any) -> np.ndarray[Any, Any]:
    network.validate_model(model)
    values = np.asarray(features, dtype=np.float64)
    if values.shape != (134,) or not np.isfinite(values).all():
        raise ValueError("finite causal neural observation required")
    values = np.clip((values - np.asarray(model["mean"])) / np.asarray(model["scale"]), -8.0, 8.0)
    layers = model["actor"]["layers"]
    for index, layer in enumerate(layers):
        values = np.asarray(layer["weight"]) @ values + np.asarray(layer["bias"])
        if index < len(layers) - 1:
            values = np.tanh(values)
    return np.asarray(values, dtype=np.float64)


def latent_sample(
    model: dict[str, Any], features: Any, frame: int
) -> tuple[np.ndarray[Any, Any], float]:
    sampling = model["training_sampling"]
    if type(frame) is not int or not 30 <= frame < 3000:
        raise ValueError("sample must follow causal decision boundary")
    mean = raw_actor_mean(model, features)
    rng = np.random.default_rng(np.random.SeedSequence(sampling["seed"], spawn_key=(frame,)))
    noise = rng.normal(size=12)
    std = sampling["std_raw"]
    sample = mean + std * noise
    log_probability = float(np.sum(-0.5 * noise**2 - np.log(std) - 0.5 * np.log(2 * np.pi)))
    return sample, log_probability


def delta_at_frame(
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
    model = policy["step_motor_proof"]["model"]
    if make_preview(model) != policy:
        raise ValueError("stochastic preview differs from sealed model and sampling law")
    features = features_at_frame(
        body,
        frame=frame,
        nominal_target=nominal_target,
        previous=previous,
        previous_contact_forces=previous_contact_forces,
    )
    base, bounds, prior = [np.asarray(v, dtype=np.float64) for v in (baseline, limits, previous)]
    if (
        base.shape != (12,)
        or bounds.shape != (12, 2)
        or prior.shape != (12,)
        or not all(np.isfinite(v).all() for v in (base, bounds, prior))
        or np.any(bounds[:, 0] >= bounds[:, 1])
        or np.max(np.abs(prior)) > CAP_RAD + 1e-5
    ):
        raise ValueError("bounded measured stochastic motor boundary required")
    if frame < 30:
        if np.any(prior != 0):
            raise ValueError("exploration cannot precede causal decision boundary")
        return np.zeros(12)
    sample, _ = latent_sample(model, features, frame)
    desired = CAP_RAD * np.tanh(sample)
    proposed = prior + np.clip(desired - prior, -SLEW_RAD, SLEW_RAD)
    lower = np.maximum(np.minimum(0.0, bounds[:, 0] - base), -CAP_RAD)
    upper = np.minimum(np.maximum(0.0, bounds[:, 1] - base), CAP_RAD)
    return np.asarray(np.clip(proposed, lower, upper), dtype=np.float64)
