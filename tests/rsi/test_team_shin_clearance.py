"""Measured shin clearance is a finite, immutable, same-player observation."""

import pytest

from rosclaw_soccer.skills.team.shin_clearance import TeamShinClearance


def test_same_player_finite_shin_differential():
    value = TeamShinClearance("red.finisher", 25, (0.01, 0.04), ((0.1,) * 6, (0.2,) * 6))
    assert value.clearance_m == (0.01, 0.04)
    with pytest.raises(ValueError, match="finite same-player"):
        TeamShinClearance("red.finisher", 25, (float("nan"), 0.04), value.gradient_m_per_rad)
    with pytest.raises(ValueError, match="finite same-player"):
        TeamShinClearance("red.finisher", 25, value.clearance_m, ((0.0,) * 5, (0.0,) * 6))
    with pytest.raises(ValueError, match="finite same-player"):
        TeamShinClearance("Red Finisher", 25, value.clearance_m, value.gradient_m_per_rad)
