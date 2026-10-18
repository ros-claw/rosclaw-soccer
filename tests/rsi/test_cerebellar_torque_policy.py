import copy

import numpy as np
import pytest

from rosclaw_soccer.rsi.cerebellar_torque_policy import (
    CAPS,
    CerebellarTorquePolicy,
    features,
    initialize,
    validate_model,
)
from rosclaw_soccer.sim.contracts import hash_json

WORLD = "sha256:" + "a" * 64


def state():
    q, v = np.zeros(43), np.zeros(41)
    q[3] = 1
    return q, v, np.zeros(29), np.zeros(29)


def reseal(model):
    model["model_hash"] = hash_json({k: v for k, v in model.items() if k != "model_hash"})


@pytest.mark.parametrize("seed", [2026101801, 2026101802, 2026101803])
def test_three_initializations_have_zero_torque_head_not_fake_learning(seed):
    model = initialize(seed=seed, world_hash=WORLD)
    policy = CerebellarTorquePolicy(model)
    q, v, target, previous = state()
    q[7:36] = np.linspace(-0.5, 0.5, 29)
    v[:35] = np.linspace(-4, 4, 35)
    np.testing.assert_array_equal(policy.offset(q, v, target, previous), np.zeros(29))
    assert model["optimizer_steps"] == policy.optimizer_steps == 0
    assert policy.world_hash == WORLD
    assert model["runtime_execution_authorized"] is False
    assert model["hardware_authorized"] is False


def test_features_use_current_proprioception_and_explicit_applied_history_only():
    inputs = state()
    before = features(*inputs)
    assert before.shape == (97,)
    q, v, target, previous = (value.copy() for value in inputs)
    q[:3] = [5, 6, 7]  # no world translation or privileged root height input
    q[36:] = np.linspace(1, 7, 7)  # ball state is not a feature
    v[35:] = np.linspace(1, 6, 6)
    np.testing.assert_array_equal(features(q, v, target, previous), before)
    q[3:7] *= -1
    np.testing.assert_array_equal(features(q, v, target, previous), before)
    previous[4] = CAPS[4] / 2
    assert features(q, v, target, previous)[-25] == 0.5
    np.testing.assert_array_equal(inputs[0], state()[0])


def test_compiled_weights_are_owned_and_numpy_matches_independent_torch_math():
    import torch

    model = initialize(seed=42, world_hash=WORLD)
    rng = np.random.default_rng(10)
    model["layers"][-1]["weight"] = rng.normal(0, 0.1, (29, 64)).tolist()
    reseal(model)
    policy = CerebellarTorquePolicy(model)
    q, v, target, previous = state()
    q[7:36] = rng.normal(0, 0.2, 29)
    v[:35] = rng.normal(size=35)
    expected_features = np.concatenate(
        (
            -q[7:36] / 0.5,
            v[6:35] / 10,
            q[3:7],
            v[:6] / [2, 2, 2, 3, 3, 3],
            previous / CAPS,
        )
    )
    value = torch.tensor(expected_features, dtype=torch.float64)
    for layer in model["layers"]:
        value = torch.tanh(
            torch.nn.functional.linear(
                value,
                torch.tensor(layer["weight"], dtype=torch.float64),
                torch.tensor(layer["bias"], dtype=torch.float64),
            )
        )
    expected = value.numpy() * CAPS
    np.testing.assert_allclose(policy.offset(q, v, target, previous), expected, atol=1e-14, rtol=0)
    model["layers"][-1]["weight"] = np.zeros((29, 64)).tolist()
    np.testing.assert_allclose(policy.offset(q, v, target, previous), expected, atol=1e-14, rtol=0)
    assert np.all(np.abs(expected) <= CAPS)


@pytest.mark.parametrize("index", range(4))
def test_nonfinite_or_wrong_dtype_observations_rejected(index):
    inputs = list(state())
    inputs[index] = inputs[index].astype(np.float32)
    with pytest.raises(ValueError, match="float64"):
        features(*inputs)
    inputs = list(state())
    inputs[index].flat[0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        features(*inputs)


def test_bad_quaternion_and_out_of_cap_history_are_not_repaired():
    q, v, target, previous = state()
    q[3] = 0.8
    with pytest.raises(ValueError, match="quaternion"):
        features(q, v, target, previous)
    q[3] = 1
    previous[0] = CAPS[0] + 0.01
    with pytest.raises(ValueError, match="actually applied"):
        features(q, v, target, previous)


@pytest.mark.parametrize("corruption", ["seal", "shape", "nan", "hardware", "caps", "world"])
def test_model_corruption_or_authority_escalation_rejected(corruption):
    model = copy.deepcopy(initialize(seed=42, world_hash=WORLD))
    if corruption == "seal":
        model["layers"][0]["weight"][0][0] += 0.01
    elif corruption == "shape":
        model["layers"][0]["weight"].pop()
    elif corruption == "nan":
        model["layers"][0]["weight"][0][0] = np.nan
    elif corruption == "hardware":
        model["hardware_authorized"] = True
    elif corruption == "caps":
        model["offset_caps_nm"][0] = 10.0
    else:
        model["world_hash"] = "unbound"
    if corruption not in ("seal", "nan"):
        reseal(model)
    with pytest.raises(ValueError):
        validate_model(model)
