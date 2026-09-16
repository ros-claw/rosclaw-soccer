import dataclasses

import pytest

from rosclaw_soccer.skills.planar_interception import (
    PlanarInterceptionConfig,
    propose_planar_interception,
)

torch = pytest.importorskip("torch")


def config():
    return PlanarInterceptionConfig((0.24, 0.0), 3.0, 0.4, 0.012)


def values():
    return dict(
        target_position_xy=torch.tensor([-0.4, 0.1]),
        target_velocity_xy=torch.tensor([0.8, 0.0]),
        effector_position_xy=torch.tensor([-0.01, 0.12]),
        previous_velocity_xy=torch.zeros(2),
    )


def test_matches_experimental_formula_exactly_without_mutating_inputs():
    inputs = values()
    before = {k: v.clone() for k, v in inputs.items()}
    position = inputs["target_position_xy"].clone()
    position[0] += 0.24
    expected = inputs["target_velocity_xy"] + 3.0 * (position - inputs["effector_position_xy"])
    expected *= (0.4 / torch.linalg.vector_norm(expected).clamp_min(1e-9)).clamp_max(1)
    expected -= inputs["previous_velocity_xy"]
    expected *= (0.012 / torch.linalg.vector_norm(expected).clamp_min(1e-9)).clamp_max(1)
    expected += inputs["previous_velocity_xy"]
    result = propose_planar_interception(**inputs, config=config())
    assert torch.equal(result, expected)
    assert all(torch.equal(v, before[k]) for k, v in inputs.items())
    result.fill_(99)
    assert all(torch.equal(v, before[k]) for k, v in inputs.items())


def test_speed_and_per_call_change_remain_bounded_across_reversals():
    previous = torch.zeros(2)
    for i in range(200):
        target = torch.tensor([10.0, -5.0]) * (-1 if (i // 50) % 2 else 1)
        result = propose_planar_interception(
            target_position_xy=target,
            target_velocity_xy=torch.zeros(2),
            effector_position_xy=torch.zeros(2),
            previous_velocity_xy=previous,
            config=config(),
        )
        assert torch.linalg.vector_norm(result) <= 0.400001
        assert torch.linalg.vector_norm(result - previous) <= 0.012001
        previous = result


@pytest.mark.parametrize("fault", ["nan", "inf", "shape", "dtype", "grad", "bound", "history"])
def test_invalid_observations_rejected_without_mutation(fault):
    inputs = values()
    if fault in ("nan", "inf"):
        inputs["target_velocity_xy"][0] = float(fault)
    elif fault == "shape":
        inputs["target_velocity_xy"] = torch.zeros(3)
    elif fault == "dtype":
        inputs["target_velocity_xy"] = torch.zeros(2, dtype=torch.float64)
    elif fault == "grad":
        inputs["target_velocity_xy"].requires_grad_(True)
    elif fault == "bound":
        inputs["target_velocity_xy"][0] = 1001
    else:
        inputs["previous_velocity_xy"][0] = 0.5
    before = inputs["previous_velocity_xy"].clone()
    with pytest.raises(ValueError):
        propose_planar_interception(**inputs, config=config())
    assert torch.equal(inputs["previous_velocity_xy"], before)


@pytest.mark.parametrize(
    "field,value",
    [
        ("position_gain_per_sec", True),
        ("position_gain_per_sec", float("nan")),
        ("position_gain_per_sec", 0),
        ("maximum_speed_mps", -1),
        ("maximum_delta_mps_per_call", 0.5),
        ("desired_offset_xy_m", [0.24, 0.0]),
        ("desired_offset_xy_m", (0.0, float("inf"))),
        ("activation_ceiling", "REAL"),
    ],
)
def test_invalid_configuration_rejected(field, value):
    with pytest.raises(ValueError):
        dataclasses.replace(config(), **{field: value})


def test_each_control_parameter_changes_commitment():
    original = config()
    for field, value in (
        ("desired_offset_xy_m", (0.25, 0.0)),
        ("position_gain_per_sec", 3.1),
        ("maximum_speed_mps", 0.3),
        ("maximum_delta_mps_per_call", 0.01),
    ):
        assert dataclasses.replace(original, **{field: value}).config_hash != original.config_hash
