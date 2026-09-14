"""Numerical contract tests; these do not certify physical shooting ability."""

import numpy as np
import pytest

from rosclaw_soccer.providers.g1.fast_contact_reflex import (
    G1FastContactReflex,
    fast_contact_features,
    fast_contact_numeric_hash,
    fast_contact_weight_shapes,
)

HASH = "sha256:" + "a" * 64


def weights():
    rng = np.random.default_rng(12)
    result = {
        key: rng.normal(0, 0.03, shape) for key, shape in fast_contact_weight_shapes().items()
    }
    result["center"] = np.zeros(81)
    result["scale"] = np.ones(81)
    result["output_mask"] = np.ones(29)
    return result


def model(parameters=None, agent="red.finisher"):
    parameters = weights() if parameters is None else parameters
    return G1FastContactReflex(
        parameters,
        agent_id=agent,
        expected_numeric_hash=fast_contact_numeric_hash(parameters),
        body_hash=HASH,
        observation_contract_hash=HASH,
    )


def observation():
    return np.zeros(81)


def test_numeric_inference_matches_explicit_formula_and_is_owned():
    parameters = weights()
    runtime = model(parameters)
    obs = observation()
    value = obs.copy()
    for layer in (0, 2, 4):
        value = value @ parameters[f"{layer}.weight"].T + parameters[f"{layer}.bias"]
        if layer != 4:
            value = np.tanh(value)
    expected = np.clip(value * 80, -120, 120)
    for parameter in parameters.values():
        parameter.fill(0)
    result = runtime.propose(agent_id="red.finisher", observation=obs)
    np.testing.assert_array_equal(result.residual_torque_nm, expected)
    np.testing.assert_array_equal(obs, observation())
    assert result.activation_ceiling == "SIM_ONLY"
    assert result.contract_hash == runtime.contract_hash


@pytest.mark.parametrize("phase", [0, 245, 267, 100000])
def test_outside_phase_envelope_is_zero(phase):
    obs = observation()
    obs[0] = (phase - 256) / 10
    assert (
        model().propose(agent_id="red.finisher", observation=obs).residual_torque_nm == (0.0,) * 29
    )


def test_output_mask_and_torque_cap():
    parameters = weights()
    parameters["4.bias"].fill(1000)
    parameters["output_mask"][0] = 0
    result = model(parameters).propose(agent_id="red.finisher", observation=observation())
    assert result.residual_torque_nm == (0.0,) + (120.0,) * 28


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf, 10001])
def test_rejects_bad_observation_even_outside_window(bad):
    obs = observation()
    obs[0] = -25
    obs[12] = bad
    with pytest.raises(ValueError):
        model().propose(agent_id="red.finisher", observation=obs)


def test_player_binding_and_semantic_inputs():
    runtime = model()
    assert runtime.contract_hash != model(agent="blue.finisher").contract_hash
    with pytest.raises(ValueError):
        runtime.propose(agent_id="blue.finisher", observation=observation())
    for index, value in ((0, 0.001), (0, -25.7), (1, 0.5)):
        obs = observation()
        obs[index] = value
        with pytest.raises(ValueError):
            runtime.propose(agent_id="red.finisher", observation=obs)
    with pytest.raises(ValueError):
        runtime.propose(agent_id="red.finisher", observation=observation().astype(np.float32))


@pytest.mark.parametrize("kind", ["extra", "missing", "dtype", "shape", "nan", "scale", "mask"])
def test_weight_contract_rejects_bad_inputs(kind):
    parameters = weights()
    if kind == "extra":
        parameters["extra"] = np.zeros(1)
    elif kind == "missing":
        del parameters["0.bias"]
    elif kind == "dtype":
        parameters["0.bias"] = parameters["0.bias"].astype(np.float32)
    elif kind == "shape":
        parameters["0.bias"] = np.zeros(127)
    elif kind == "nan":
        parameters["0.bias"][0] = np.nan
    elif kind == "scale":
        parameters["scale"][0] = 0.099
    else:
        parameters["output_mask"][0] = 0.5
    with pytest.raises(ValueError):
        fast_contact_numeric_hash(parameters)


def test_wrong_numeric_hash_and_hash_format():
    for digest in (HASH, "sha256:" + "g" * 64):
        with pytest.raises(ValueError):
            G1FastContactReflex(
                weights(),
                agent_id="red.finisher",
                expected_numeric_hash=digest,
                body_hash=HASH,
                observation_contract_hash=HASH,
            )


def feature_arguments():
    return dict(
        policy_frame=256,
        contact_latched=True,
        joint_position=np.ones(29),
        joint_velocity=np.ones(29) * 10,
        ball_minus_foot_world=np.ones(3),
        ball_velocity_world=np.ones(3) * 10,
        root_velocity=np.ones(6) * 5,
        foot_spatial_velocity=np.ones(6) * 10,
        goal_minus_ball_world=np.ones(3) * 10,
    )


def test_feature_order_units_and_no_aliasing():
    arguments = feature_arguments()
    result = fast_contact_features(**arguments)
    np.testing.assert_array_equal(result, np.r_[0.0, np.ones(80)])
    assert result.dtype == np.float64
    arguments["joint_position"].fill(0)
    np.testing.assert_array_equal(result[2:31], np.ones(29))


@pytest.mark.parametrize(
    "key,value",
    [
        ("policy_frame", True),
        ("policy_frame", 1.5),
        ("policy_frame", -1),
        ("contact_latched", 1),
        ("joint_position", np.zeros(30)),
        ("joint_velocity", np.full(29, np.nan)),
        ("ball_velocity_world", np.zeros(3, dtype=int)),
    ],
)
def test_feature_validation(key, value):
    arguments = feature_arguments()
    arguments[key] = value
    with pytest.raises(ValueError):
        fast_contact_features(**arguments)
