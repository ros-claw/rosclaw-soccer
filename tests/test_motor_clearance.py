from dataclasses import replace

import numpy as np
import pytest

from rosclaw_soccer.skills.team.independent_team_world import IndependentTeamWorldConfig
from rosclaw_soccer.world.motor_clearance import motor_clearance_planes
from rosclaw_soccer.world.player_clearance import propose_clearance_velocity


def test_preview_adds_early_clearance_for_an_approaching_motor_peer():
    offsets = np.array([[1.1, 0.0]])
    planes = motor_clearance_planes(offsets, np.array([[-1.0, 0.0]]), horizon_sec=0.4)
    assert planes.shape == (1, 3) and planes[0, 2] > 0
    result = propose_clearance_velocity(
        np.array([0.2, 0.1]), offsets, maximum_speed_mps=0.7, additional_halfplanes=planes
    )
    assert result.feasible and result.constrained and result.velocity_mps[0] < 0
    assert result.velocity_mps[1] == 0.1


def test_stationary_or_receding_peer_does_not_change_current_clearance():
    for velocity in ((0.0, 0.0), (1.0, 0.0)):
        planes = motor_clearance_planes(
            np.array([[1.0, 0.0]]), np.array([velocity]), horizon_sec=0.4
        )
        assert planes.shape == (0, 3)


def test_prediction_is_explicit_and_preserves_legacy_hash_by_default():
    config = IndependentTeamWorldConfig()
    assert (
        config.config_hash
        == "sha256:fd442bce83f737c64e2ee59ecc09dc204376100f9476a21361b41b538d8c41d1"
    )
    for value in (True, float("nan"), -0.1, 0.6):
        with pytest.raises(ValueError):
            replace(config, all_role_clearance=True, motor_clearance_prediction_sec=value)
    with pytest.raises(ValueError):
        replace(config, motor_clearance_prediction_sec=0.3)
    assert (
        replace(config, all_role_clearance=True, motor_clearance_prediction_sec=0.3).config_hash
        != config.config_hash
    )


def test_nonfinite_preview_inputs_are_rejected():
    with pytest.raises(ValueError):
        motor_clearance_planes(np.zeros((1, 2)), np.full((1, 2), np.nan), horizon_sec=0.3)
