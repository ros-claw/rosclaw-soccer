"""Partial-observation short response model from ACTUAL measured SIM outcomes.

Future velocity is a label, never an input. This is not a Markov world model,
actor, motor skill, learned safety gate or replacement for native integration.
Generic optimizer math is reused from ROSClaw Core, not copied into Soccer.
"""

from typing import Any

import numpy as np

from rosclaw_soccer.sim.contracts import hash_json

WIDTH = 17
SEEDS = (1019, 2027, 4093)
MARGINAL_SCHEMA = "soccer.research.observed_ball_response_model.v1"
EFFECT_SCHEMA = "soccer.research.matched_ball_action_effect_model.v1"


def observation(
    *,
    frame: int,
    side: int,
    foot_xyz: np.ndarray,
    previous_foot_xyz: np.ndarray,
    ball_xyz: np.ndarray,
    ball_velocity: np.ndarray,
    leg_joint_angles: np.ndarray,
    root_angular_velocity: np.ndarray,
    previous_forces: np.ndarray,
    requested_wrench_coefficients: np.ndarray,
) -> np.ndarray:
    """Current/past arrays only; reflection-canonicalize the measured foot.

    Wrench coefficients describe a requested40N/4Nm Jacobian wish, NOT a future
    measured impulse. The action can be zero even if no forecast was eligible.
    The background recurrent controller state is not fully observed here.
    """
    if (
        type(frame) is not int
        or not 31 <= frame <= 295
        or type(side) is not int
        or side not in (0, 1)
    ):
        raise ValueError("completed-contact current frame with observed80ms tail required")
    for value, width in (
        (foot_xyz, 3),
        (previous_foot_xyz, 3),
        (ball_xyz, 3),
        (ball_velocity, 3),
        (leg_joint_angles, 6),
        (root_angular_velocity, 3),
        (previous_forces, 6),
        (requested_wrench_coefficients, 4),
    ):
        if (
            type(value) is not np.ndarray
            or value.shape != (width,)
            or value.dtype != np.float64
            or not np.isfinite(value).all()
            or np.max(abs(value)) > 1e6
        ):
            raise ValueError("bounded finite current/past float64 physical arrays required")
    if (
        np.any(previous_forces < 0)
        or previous_forces[side] <= 1
        or side != int(np.argmax(previous_forces[:2]))
    ):
        raise ValueError("own completed foot force and its measured side required")
    if np.max(abs(requested_wrench_coefficients)) > 1:
        raise ValueError("bounded declared wrench wish coefficients required")
    sign = 1.0 if side == 0 else -1.0
    gap = (ball_xyz - foot_xyz) * [1.0, sign, 1.0]
    velocity = (ball_velocity - (foot_xyz - previous_foot_xyz) / 0.02) * [1.0, sign, 1.0]
    state = [
        np.log1p(previous_forces[side]),
        np.log1p(previous_forces[1 - side]),
        leg_joint_angles[4],
        sign * leg_joint_angles[5],
        sign * root_angular_velocity[0],
        root_angular_velocity[1],
    ]
    action = requested_wrench_coefficients * np.asarray([1.0, sign, 1.0, sign])
    result = np.concatenate((gap, velocity, state, action, [(20 - frame % 20) / 20])).astype(
        np.float64
    )
    if result.shape != (WIDTH,) or not np.isfinite(result).all():
        raise ValueError("finite17-feature partial measured response observation required")
    return result


def _features(value: np.ndarray) -> None:
    if (
        type(value) is not np.ndarray
        or value.dtype != np.float64
        or value.ndim != 2
        or value.shape[1] != WIDTH
        or not 1 <= len(value) <= 4096
        or not np.isfinite(value).all()
        or np.max(abs(value)) > 1e6
    ):
        raise ValueError("finite bounded float64 response feature matrix required")


def fit(
    features: np.ndarray, delta_velocity: np.ndarray, *, seed: int, steps: int = 256
) -> dict[str, Any]:
    """Three independent scalar Core regressors, m/s delta targets, no actor.

    Caller supplies ONLY the declared training source clusters. Source integrity,
    validation/test scoring and physical interpretation are outside this function.
    """
    return _fit_heads(features, delta_velocity, seed=seed, steps=steps, paired_rows=None)


def fit_matched_effect(
    baseline_features: np.ndarray,
    action_features: np.ndarray,
    baseline_delta_velocity: np.ndarray,
    action_delta_velocity: np.ndarray,
    *,
    seed: int,
    steps: int = 256,
) -> dict[str, Any]:
    """Fit factual paired effects with explicit mathematical zero identities.

    The caller must prove full matched initial/controller/noise histories. Partial
    observation equality alone does not prove a causal experiment. Identity anchors
    are NOT additional physical trajectories or independent causal examples.
    """
    for x in (baseline_features, action_features):
        _features(x)
    if (
        len(baseline_features) > 2048
        or action_features.shape != baseline_features.shape
        or np.any(baseline_features[:, 12:16])
        or not np.all(np.any(action_features[:, 12:16] != 0, axis=1))
        or np.max(abs(action_features[:, 12:16])) > 1
        or not np.array_equal(baseline_features[:, :12], action_features[:, :12])
        or not np.array_equal(baseline_features[:, 16], action_features[:, 16])
    ):
        raise ValueError("matched current state/clock and nonzero-versus-zero action required")
    for target in (baseline_delta_velocity, action_delta_velocity):
        if (
            type(target) is not np.ndarray
            or target.dtype != np.float64
            or target.shape != (len(baseline_features), 3)
            or not np.isfinite(target).all()
            or np.max(abs(target)) > 1e6
        ):
            raise ValueError("finite actual paired velocity labels required")
    effect = action_delta_velocity - baseline_delta_velocity
    return _fit_heads(
        np.concatenate((action_features, baseline_features)),
        np.concatenate((effect, np.zeros_like(effect))),
        seed=seed,
        steps=steps,
        paired_rows=len(effect),
    )


def _fit_heads(
    features: np.ndarray,
    delta_velocity: np.ndarray,
    *,
    seed: int,
    steps: int,
    paired_rows: int | None,
) -> dict[str, Any]:
    _features(features)
    if (
        type(delta_velocity) is not np.ndarray
        or delta_velocity.dtype != np.float64
        or delta_velocity.shape != (len(features), 3)
        or not np.isfinite(delta_velocity).all()
        or np.max(abs(delta_velocity)) > 1e6
        or type(seed) is not int
        or seed not in SEEDS
        or type(steps) is not int
        or not 1 <= steps <= 256
    ):
        raise ValueError(
            "finite measured training targets and bounded declared seed/steps required"
        )
    from rosclaw.growth.normalized_value_regression import (
        NormalizedValueRegressionConfig,
        fit_normalized_value_regression,
    )

    center = features.mean(axis=0)
    scale = np.maximum(features.std(axis=0), 0.01)
    x = np.clip((features - center) / scale, -10.0, 10.0)
    heads = []
    for dimension in range(3):
        rng = np.random.default_rng(seed + dimension * 17)
        parameters = dict(
            weight_0=(rng.normal(size=(64, WIDTH)) * 0.05).tolist(),
            bias_0=np.zeros(64).tolist(),
            weight_1=np.zeros((1, 64)).tolist(),
            bias_1=[0.0],
        )
        result = fit_normalized_value_regression(
            context=x[:, None, :],
            value_targets=delta_velocity[:, dimension, None],
            critic_parameters=parameters,
            config=NormalizedValueRegressionConfig(
                steps=steps,
                batch_episodes=min(64, len(x)),
                learning_rate=0.0003,
                seed=seed + dimension * 17,
            ),
        )
        assert (
            result["accepted_value_optimizer_steps"] == steps
            and result["actor_optimizer_updates"] == 0
        )
        heads.append(result)
    model = dict(
        schema=MARGINAL_SCHEMA if paired_rows is None else EFFECT_SCHEMA,
        mode="SIM_ONLY",
        seed=seed,
        width=WIDTH,
        center=center.tolist(),
        scale=scale.tolist(),
        heads=heads,
        optimizer_steps=sum(h["accepted_value_optimizer_steps"] for h in heads),
        target="mirrored_actual_cached_ball_velocity_tplus4_minus_t_m_s"
        if paired_rows is None
        else "mirrored_actual_paired_total_controller_conditioned_action_effect_80ms_m_s",
        native_trajectory_verified_by_caller=False,
        partial_observation_not_markov=True,
        actor_optimizer_updates=0,
        used_in_control=False,
        teacher_qualified=False,
        runtime_execution_authorized=False,
        promotion_authorized=False,
        hardware_authorized=False,
    )
    if paired_rows is not None:
        model.update(
            training_matched_physical_pairs=paired_rows,
            mathematical_zero_identity_rows=paired_rows,
            identity_rows_are_new_physical_examples=False,
            full_matched_history_verified_by_caller=False,
            prediction_transform="raw(s,a)-raw(s,0)",
        )
    model["model_hash"] = hash_json(model)
    return model


def predict(model: dict[str, Any], features: np.ndarray) -> np.ndarray:
    return _predict_raw(model, features, schema=MARGINAL_SCHEMA)


def predict_matched_effect(model: dict[str, Any], features: np.ndarray) -> np.ndarray:
    """Exact zero-action boundary, no future data or action selection API."""
    _features(features)
    if (
        np.max(abs(features[:, 12:16])) > 1
        or model.get("prediction_transform") != "raw(s,a)-raw(s,0)"
    ):
        raise ValueError("bounded matched-effect wish and exact zero-boundary transform required")
    zero = features.copy()
    zero[:, 12:16] = 0
    result: np.ndarray = _predict_raw(model, features, schema=EFFECT_SCHEMA) - _predict_raw(
        model, zero, schema=EFFECT_SCHEMA
    )
    if not np.isfinite(result).all():
        raise ValueError("finite matched action-effect difference required")
    return result


def _predict_raw(model: dict[str, Any], features: np.ndarray, *, schema: str) -> np.ndarray:
    _features(features)
    if (
        model.get("schema") != schema
        or model.get("model_hash")
        != hash_json({k: v for k, v in model.items() if k != "model_hash"})
        or model.get("width") != WIDTH
        or model.get("mode") != "SIM_ONLY"
        or len(model.get("heads", [])) != 3
        or any(
            model.get(k) is not False
            for k in (
                "used_in_control",
                "teacher_qualified",
                "runtime_execution_authorized",
                "promotion_authorized",
                "hardware_authorized",
            )
        )
    ):
        raise ValueError("sealed non-controlling SIM response model required")
    center = np.asarray(model["center"], dtype=np.float64)
    scale = np.asarray(model["scale"], dtype=np.float64)
    if (
        center.shape != (WIDTH,)
        or scale.shape != (WIDTH,)
        or not np.isfinite(center).all()
        or not np.isfinite(scale).all()
        or np.any(scale < 0.01)
    ):
        raise ValueError("finite training-only normalization required")
    x = np.clip((features - center) / scale, -10.0, 10.0)
    outputs = []
    for head in model["heads"]:
        params = {k: np.asarray(v, dtype=np.float64) for k, v in head["critic_parameters"].items()}
        shapes = dict(weight_0=(64, WIDTH), bias_0=(64,), weight_1=(1, 64), bias_1=(1,))
        if params.keys() != shapes.keys() or any(
            params[k].shape != shape or not np.isfinite(params[k]).all()
            for k, shape in shapes.items()
        ):
            raise ValueError("finite complete original-unit Core scalar heads required")
        hidden = np.tanh(x @ params["weight_0"].T + params["bias_0"])
        outputs.append((hidden @ params["weight_1"].T + params["bias_1"])[:, 0])
    result = np.stack(outputs, axis=1)
    if not np.isfinite(result).all():
        raise ValueError("finite partial response prediction required")
    return result
