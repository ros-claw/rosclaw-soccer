"""Low-ball admission is opt-in; success and collision geometry are unchanged."""

from dataclasses import asdict

import numpy as np
import pytest

from rosclaw_soccer.providers.g1.shared_keeper_reach import SharedKeeperReachConfig
from rosclaw_soccer.sim.contracts import hash_json


@pytest.mark.parametrize(
    "height",
    [True, False, np.bool_(True), None, "0.3", float("nan"), float("inf"), -0.1, 0.19, 0.651],
)
def test_invalid_height_is_rejected(height):
    with pytest.raises(ValueError, match="interception height"):
        SharedKeeperReachConfig(minimum_intercept_height_m=height)


@pytest.mark.parametrize("height", [0.20, 0.30, 0.50, 0.65])
def test_height_is_explicitly_bound_to_config(height):
    config = SharedKeeperReachConfig(minimum_intercept_height_m=height)
    assert asdict(config)["minimum_intercept_height_m"] == height
    assert config.minimum_intercept_height_m == height
    if height != 0.65:
        assert hash_json(asdict(config)) != hash_json(asdict(SharedKeeperReachConfig()))


def test_legacy_height_remains_default():
    assert SharedKeeperReachConfig().minimum_intercept_height_m == 0.65
    assert SharedKeeperReachConfig().ballistic_airborne_velocity is False
