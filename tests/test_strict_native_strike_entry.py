from dataclasses import replace

import pytest

from rosclaw_soccer.growth.locomotion_contact_teacher import G1RollingOptionBridgeConfig


@pytest.mark.parametrize("tolerance", [0.05, 0.10, 0.12, 0.15])
def test_more_precise_native_entry_can_be_requested_without_changing_defaults(tolerance):
    old = G1RollingOptionBridgeConfig()
    assert old.maximum_strike_lateral_error_m == 0.50
    strict = replace(old, maximum_strike_lateral_error_m=tolerance)
    assert strict.config_hash != old.config_hash
    assert strict.maximum_strike_yaw_error_rad == old.maximum_strike_yaw_error_rad
    assert strict.minimum_strike_stance_depth_m == old.minimum_strike_stance_depth_m
    assert strict.activation_ceiling == "SIM_ONLY"
    assert strict.shoot_parameters == old.shoot_parameters


@pytest.mark.parametrize("tolerance", [0.049, 0.601, float("nan"), float("inf")])
def test_lateral_entry_still_has_a_bounded_validated_range(tolerance):
    with pytest.raises(ValueError):
        G1RollingOptionBridgeConfig(maximum_strike_lateral_error_m=tolerance)
