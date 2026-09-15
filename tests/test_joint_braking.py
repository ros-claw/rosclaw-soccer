import numpy as np
import pytest

from rosclaw_soccer.sim.joint_braking import strengthen_outward_joint_braking
from rosclaw_soccer.skills.team.independent_team_world import IndependentTeamWorldConfig


def arguments():
    return dict(
        joint_position=np.array([-0.95, 0.95, -0.95, 0.95, 0.0]),
        joint_velocity=np.array([-1.0, 1.0, 1.0, -1.0, 0.0]),
        projected_torque=np.array([8.0, -8.0, 7.0, -7.0, 2.0]),
        joint_ranges=np.tile(np.array([-1.0, 1.0]), (5, 1)),
        limited=np.ones(5, dtype=bool),
        selected_joints=(0, 1, 2, 3),
        damping=16.0,
    )


def test_only_outward_threats_gain_braking_without_mutating_arrays():
    args = arguments()
    original = {k: v.copy() for k, v in args.items() if isinstance(v, np.ndarray)}
    result = strengthen_outward_joint_braking(**args)
    np.testing.assert_allclose(result[:2], [15.2, -15.2])
    np.testing.assert_array_equal(result[2:], args["projected_torque"][2:])
    for key, value in original.items():
        np.testing.assert_array_equal(args[key], value)


def test_existing_stronger_torque_unselected_and_unlimited_joints_unchanged():
    args = arguments()
    args["projected_torque"][:2] = [100.0, -100.0]
    np.testing.assert_array_equal(
        strengthen_outward_joint_braking(**args), args["projected_torque"]
    )
    args = arguments()
    args["limited"][0] = False
    args["selected_joints"] = (0,)
    np.testing.assert_array_equal(
        strengthen_outward_joint_braking(**args), args["projected_torque"]
    )


@pytest.mark.parametrize(
    "fault", ["nan", "shape", "mask", "duplicate", "index", "bool_index", "damping", "narrow"]
)
def test_invalid_projection_contracts_rejected(fault):
    args = arguments()
    if fault == "nan":
        args["joint_velocity"][0] = np.nan
    elif fault == "shape":
        args["joint_ranges"] = np.zeros((2, 2))
    elif fault == "mask":
        args["limited"] = np.ones(5)
    elif fault == "duplicate":
        args["selected_joints"] = (0, 0)
    elif fault == "index":
        args["selected_joints"] = (5,)
    elif fault == "bool_index":
        args["selected_joints"] = (True,)
    elif fault == "damping":
        args["damping"] = float("inf")
    else:
        args["joint_ranges"][0] = (-0.01, 0.01)
    with pytest.raises(ValueError):
        strengthen_outward_joint_braking(**args)


def test_world_braking_opt_in_is_content_bound_and_default_stays_disabled():
    old = IndependentTeamWorldConfig()
    assert old.outward_waist_braking_damping is None
    assert (
        old.config_hash == "sha256:fd442bce83f737c64e2ee59ecc09dc204376100f9476a21361b41b538d8c41d1"
    )
    assert (
        IndependentTeamWorldConfig(outward_waist_braking_damping=16.0).config_hash
        != old.config_hash
    )
    for invalid in (True, 5.0, 21.0, float("nan")):
        with pytest.raises(ValueError):
            IndependentTeamWorldConfig(outward_waist_braking_damping=invalid)
