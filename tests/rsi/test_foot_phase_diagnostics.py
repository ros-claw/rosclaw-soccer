"""Avoid interpreting post-impact limb motion as a pre-contact phase option."""

from __future__ import annotations

import numpy as np
import pytest

from rosclaw_soccer.rsi.foot_phase_diagnostics import best_precontact_offset


def test_phase_window_never_looks_after_first_contact() -> None:
    offset, gain = best_precontact_offset(np.asarray([0.0, 0.1, 0.0, 1.0]), first=2, lookback=2)
    assert offset == -1
    assert gain == pytest.approx(0.1)


def test_phase_window_rejects_incomplete_or_nonfinite_history() -> None:
    with pytest.raises(ValueError, match="precontact"):
        best_precontact_offset(np.asarray([0.0, 0.1]), first=1, lookback=2)
    with pytest.raises(ValueError, match="precontact"):
        best_precontact_offset(np.asarray([0.0, np.nan, 0.1]), first=2, lookback=2)
