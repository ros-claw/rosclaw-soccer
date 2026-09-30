"""Fixed-horizon ball metrics are independently recomputable and fail closed."""

from __future__ import annotations

import numpy as np
import pytest

from rosclaw_soccer.rsi.independent_first_touch_bank import post_contact_displacement


def test_post_contact_displacement_uses_measured_sixty_frame_window() -> None:
    ball = np.zeros((300, 3))
    ball[140, :2] = (1.2, -0.3)
    result = post_contact_displacement(ball, 80)
    assert result["forward_60_m"] == pytest.approx(1.2)
    assert result["lateral_60_m"] == pytest.approx(-0.3)
    assert result["lateral_over_forward_60"] == pytest.approx(0.25)
    assert post_contact_displacement(ball, None)["forward_60_m"] is None
    assert post_contact_displacement(ball, 260)["lateral_60_m"] is None


def test_post_contact_displacement_rejects_malformed_trajectory() -> None:
    with pytest.raises(ValueError):
        post_contact_displacement(np.zeros((299, 3)), 80)
    ball = np.zeros((300, 3))
    ball[80, 1] = np.nan
    with pytest.raises(ValueError):
        post_contact_displacement(ball, 80)
    with pytest.raises(ValueError):
        post_contact_displacement(np.zeros((300, 3)), True)
