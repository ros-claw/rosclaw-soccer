from __future__ import annotations

from dataclasses import replace

import pytest

from rosclaw_soccer.training.second_striker_contact_exam import (
    SecondStrikerContactExamConfig,
)


def test_second_striker_contact_contract_is_bounded_and_honest() -> None:
    config = SecondStrikerContactExamConfig()
    assert config.activation_ceiling == "SIM_ONLY"
    assert not config.hardware_authorized
    assert not config.commercial_use_allowed
    assert config.maximum_torque_fraction == pytest.approx(0.85)
    with pytest.raises(ValueError, match="SIM_ONLY"):
        replace(config, hardware_authorized=True)
    with pytest.raises(ValueError, match="strike pocket"):
        replace(config, second_ball_origin_m=(2.0, 0.0, 0.115))
    with pytest.raises(ValueError, match="torque envelope"):
        replace(config, maximum_torque_fraction=0.95)
    with pytest.raises(ValueError, match="maximum contact-force"):
        replace(config, maximum_contact_force_n=100.0)
