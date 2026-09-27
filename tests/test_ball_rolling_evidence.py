from __future__ import annotations

import numpy as np
import pytest

from rosclaw_soccer.sim.ball_rolling_evidence import precontact_rolling_evidence


def test_pure_roll_and_preimpact_cutoff() -> None:
    position = np.tile([0.0, 0.0, 0.11], (20, 1))
    velocity = np.tile([-0.5, 0.0, 0.0], (20, 1))
    angular = np.tile([0.0, -0.5 / 0.11, 0.0], (20, 1))
    angular[12:, 1] = 0.0
    result = precontact_rolling_evidence(
        position, velocity, angular, first_body_impact_microstep=120
    )
    assert result.grounded_frames == 12
    assert result.mean_slip_m_s == pytest.approx(0.0)
    assert result.maximum_slip_m_s == pytest.approx(0.0)


def test_sliding_or_airborne_does_not_claim_rolling() -> None:
    position = np.tile([0.0, 0.0, 0.11], (20, 1))
    velocity = np.tile([-0.5, 0.0, 0.0], (20, 1))
    angular = np.zeros((20, 3))
    sliding = precontact_rolling_evidence(
        position, velocity, angular, first_body_impact_microstep=None
    )
    assert sliding.mean_slip_m_s == pytest.approx(0.5)
    position[:, 2] = 0.5
    airborne = precontact_rolling_evidence(
        position, velocity, angular, first_body_impact_microstep=None
    )
    assert airborne.grounded_frames == 0
    assert airborne.maximum_slip_m_s is None


def test_rejects_nonfinite_and_misaligned_observations() -> None:
    position = np.zeros((20, 3))
    with pytest.raises(ValueError):
        precontact_rolling_evidence(
            position, np.zeros((19, 3)), np.zeros((20, 3)), first_body_impact_microstep=None
        )
    position[0, 0] = np.nan
    with pytest.raises(ValueError):
        precontact_rolling_evidence(
            position, np.zeros((20, 3)), np.zeros((20, 3)), first_body_impact_microstep=None
        )
