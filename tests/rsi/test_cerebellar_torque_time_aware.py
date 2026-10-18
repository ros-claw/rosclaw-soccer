import numpy as np
import pytest

from rosclaw_soccer.rsi.cerebellar_torque_policy import (
    CAPS,
    TIME_AWARE_OBSERVATION,
    TIME_AWARE_SCHEMA,
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
    return [q, v, np.zeros(29), np.zeros(29)]


def reseal(model):
    model["model_hash"] = hash_json({k: v for k, v in model.items() if k != "model_hash"})


def test_default_initialization_preserves_exact_frozen_v1_model():
    # Independently obtained from frozen 512cd75, not from the new producer.
    model = initialize(seed=42, world_hash=WORLD)
    assert (
        model["model_hash"]
        == "sha256:9e6d2dd5cb69a85ab95f53fd61f8f99b816360741316526ecfe2094503887d47"
    )


def test_appends_actual_joint_pose_and_clock_without_repacking_v1():
    inputs = state()
    inputs[0][7:36] = np.linspace(-1, 1, 29)
    old = features(*inputs)
    new = features(*inputs, substep_index=750)
    assert new.shape == (127,)
    np.testing.assert_array_equal(new[:97], old)
    np.testing.assert_array_equal(new[97:126], inputs[0][7:36] / 2)
    assert new[126] == 0.75
    inputs[0][7:36] += 0.1
    inputs[2] += 0.1
    changed = features(*inputs, substep_index=750)
    np.testing.assert_allclose(changed[:97], old, rtol=0, atol=1e-15)
    assert not np.array_equal(changed[97:126], new[97:126])


def test_clock_is_not_ball_truth_or_future_outcome():
    inputs = state()
    expected = features(*inputs, substep_index=2999)
    assert expected[-1] == 1 / 3000
    inputs[0][:3] = [100, 200, 300]
    inputs[0][36:] = np.arange(7)
    inputs[1][35:] = np.arange(6)
    np.testing.assert_array_equal(features(*inputs, substep_index=2999), expected)


@pytest.mark.parametrize("step", [-1, 3000, True, 1.0, np.int64(1)])
def test_clock_requires_exact_before_integration_step(step):
    with pytest.raises(ValueError, match="substep index"):
        features(*state(), substep_index=step)


def test_version_cannot_silently_change_input_semantics_or_consume_bad_draw():
    v1 = CerebellarTorquePolicy(initialize(seed=42, world_hash=WORLD), sampling_seed=10)
    v2 = CerebellarTorquePolicy(
        initialize(seed=42, world_hash=WORLD, time_aware=True), sampling_seed=10
    )
    with pytest.raises(ValueError, match="clock"):
        v1.offset(*state(), substep_index=0)
    with pytest.raises(ValueError, match="clock"):
        v2.offset(*state())
    with pytest.raises(ValueError, match="substep"):
        v2.offset(*state(), substep_index=3000)
    assert v1.samples_drawn == v2.samples_drawn == 0
    np.testing.assert_array_equal(v1.offset(*state()), v2.offset(*state(), substep_index=0))


def test_nonzero_forward_matches_torch_and_responds_to_task_clock():
    import torch

    model = initialize(seed=42, world_hash=WORLD, time_aware=True)
    assert model["schema"] == TIME_AWARE_SCHEMA
    assert model["observation"] == TIME_AWARE_OBSERVATION
    assert model["optimizer_steps"] == 0
    rng = np.random.default_rng(9)
    model["layers"][-1]["weight"] = rng.normal(0, 0.1, (29, 64)).tolist()
    reseal(model)
    inputs = state()
    inputs[0][7:36] = rng.normal(0, 0.5, 29)
    inputs[1][:35] = rng.normal(size=35)
    value = torch.tensor(features(*inputs, substep_index=1500), dtype=torch.float64)
    for layer in model["layers"]:
        value = torch.tanh(
            torch.nn.functional.linear(
                value,
                torch.tensor(layer["weight"], dtype=torch.float64),
                torch.tensor(layer["bias"], dtype=torch.float64),
            )
        )
    policy = CerebellarTorquePolicy(model)
    np.testing.assert_allclose(
        policy.offset(*inputs, substep_index=1500), value.numpy() * CAPS, rtol=0, atol=1e-14
    )
    assert not np.array_equal(
        policy.offset(*inputs, substep_index=0), policy.offset(*inputs, substep_index=2999)
    )


@pytest.mark.parametrize("corruption", ["clock", "pose", "architecture", "schema"])
def test_observation_semantics_are_sealed_and_required(corruption):
    model = initialize(seed=42, world_hash=WORLD, time_aware=True)
    if corruption == "clock":
        model["observation"]["horizon_substeps"] = 4000
    elif corruption == "pose":
        model["observation"]["joint_position_scale_rad"] = 4
    elif corruption == "architecture":
        model["architecture"][0] = 97
    else:
        model["schema"] = "soccer.rsi.cerebellar_torque_residual.v1"
    reseal(model)
    with pytest.raises(ValueError, match="model"):
        validate_model(model)
