from __future__ import annotations

from dataclasses import replace

import pytest

from rosclaw_soccer.training.goalkeeper_dive_memory import GoalkeeperDiveMemoryConfig


def test_dive_memory_contract_is_content_bound_and_sim_only() -> None:
    config = GoalkeeperDiveMemoryConfig(epochs=10)
    assert config.config_hash.startswith("sha256:")
    assert config.maximum_position_rmse_rad < config.maximum_absolute_error_rad
    with pytest.raises(ValueError, match="SIM_ONLY"):
        replace(config, hardware_authorized=True)
    with pytest.raises(ValueError, match="SIM_ONLY"):
        replace(config, commercial_use_allowed=True)
    with pytest.raises(ValueError, match="epoch"):
        replace(config, epochs=9)
