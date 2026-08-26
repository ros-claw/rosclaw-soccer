from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from rosclaw_soccer.training.goalkeeper_agility import (
    GoalkeeperAgilityConfig,
    shape_goalkeeper_action_numpy,
    shape_goalkeeper_action_torch,
)


def _inputs() -> dict[str, np.ndarray]:
    action = np.zeros((3, 18), dtype=np.float64)
    action[:, 0] = (0.2, -0.3, 0.8)
    action[:, 1:] = 0.8
    return {
        "requested_action": action,
        "root_lateral_position_m": np.asarray((0.0, 0.30, -0.01)),
        "root_lateral_velocity_mps": np.asarray((0.0, 0.20, 0.0)),
        "root_angular_velocity_rad_s": np.asarray(
            ((0.0, 0.0, 0.0), (0.0, 0.0, 3.2), (0.0, 0.0, 2.35))
        ),
        "shot_active": np.asarray((True, False, True), dtype=np.bool_),
    }


def test_agility_shapes_lateral_recenters_and_preserves_leg_authority() -> None:
    desired, scale = shape_goalkeeper_action_numpy(**_inputs())
    assert desired[0, 0] > 0.2
    assert desired[1, 0] > 0.0  # positive world-y position commands local return
    assert np.count_nonzero(desired[1, 1:]) == 0
    assert desired[2, 0] > 0.8
    assert desired[2, 1] < 0.8  # yaw counter-rotation opposes positive root yaw
    assert abs(desired[0, 4]) <= 0.70 + 1.0e-12
    assert scale[0] == 1.0
    assert scale[1] < scale[2] < 1.0
    assert np.all(np.abs(desired) <= 1.0)


def test_agility_numpy_and_torch_are_equivalent() -> None:
    torch = pytest.importorskip("torch")
    inputs = _inputs()
    expected, expected_scale = shape_goalkeeper_action_numpy(**inputs)
    actual, actual_scale = shape_goalkeeper_action_torch(
        **{name: torch.from_numpy(value) for name, value in inputs.items()}
    )
    np.testing.assert_allclose(actual.numpy(), expected, atol=1e-7)
    np.testing.assert_allclose(actual_scale.numpy(), expected_scale, atol=1e-7)


def test_agility_contract_fails_closed() -> None:
    config = GoalkeeperAgilityConfig()
    assert config.config_hash.startswith("sha256:")
    with pytest.raises(ValueError, match="SIM_ONLY"):
        replace(config, hardware_authorized=True)
    with pytest.raises(ValueError, match="shapes"):
        inputs = _inputs()
        inputs["requested_action"] = np.zeros((3, 17), dtype=np.float64)
        shape_goalkeeper_action_numpy(**inputs)
