from __future__ import annotations

from dataclasses import replace

import pytest

from rosclaw_soccer.growth.upper_corner_strike import (
    UpperCornerLaneAction,
    UpperCornerStrikePolicy,
)
from rosclaw_soccer.training.upper_corner_strike_evidence import (
    UpperCornerStrikeEvidenceConfig,
)


def test_upper_corner_policy_is_bilateral_bounded_and_hashable() -> None:
    policy = UpperCornerStrikePolicy()

    assert policy.action("left-post").foot_yaw_offset_rad == pytest.approx(0.005)
    assert policy.action("right-post").foot_yaw_offset_rad == pytest.approx(-0.0162)
    assert policy.torque_config().right_leg_residual_nm == (
        5.0,
        0.0,
        0.0,
        0.0,
        -4.0,
        5.0,
    )
    assert policy.artifact_hash.startswith("sha256:")
    assert not policy.hardware_authorized


def test_upper_corner_policy_rejects_a_missing_lane() -> None:
    with pytest.raises(ValueError, match="both regulation lanes"):
        UpperCornerStrikePolicy(actions=(UpperCornerLaneAction("left-post", 0.0),))


def test_upper_corner_policy_reuses_torque_safety_contract() -> None:
    with pytest.raises(ValueError, match="exceeds"):
        UpperCornerStrikePolicy(right_leg_residual_nm=(13.0, 0.0, 0.0, 0.0, 0.0, 0.0))


def test_upper_corner_holdout_must_be_disjoint_from_discovery() -> None:
    with pytest.raises(ValueError, match="disjoint"):
        UpperCornerStrikeEvidenceConfig(sealed_holdout_friction=(0.0975, 0.10))


def test_upper_corner_config_cannot_relax_below_upper_region() -> None:
    with pytest.raises(ValueError, match="upper region"):
        UpperCornerStrikeEvidenceConfig(minimum_crossing_height_m=1.4)


def test_upper_corner_policy_cannot_authorize_hardware() -> None:
    policy = UpperCornerStrikePolicy()

    with pytest.raises(ValueError, match="SIM_ONLY"):
        replace(policy, hardware_authorized=True)
