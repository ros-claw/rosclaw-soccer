from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from rosclaw_soccer.training.second_striker_contact_exam import (
    SecondStrikerContactExamConfig,
    validate_second_striker_contact_exam,
)

_EVIDENCE = Path(
    "/code/rosclaw/rosclaw_football/evidence/athlete-foundation-v1/"
    "s108-fourth-g1-second-ball-contact-v2/evidence.json"
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


def test_external_second_striker_contact_evidence_if_available() -> None:
    if not _EVIDENCE.is_file():
        pytest.skip("external S108 physics evidence is not present")
    payload = validate_second_striker_contact_exam(_EVIDENCE)
    assert payload["passed"] is True
    assert payload["strict_replay"] is True
    assert payload["complete_second_save_claimed"] is False
