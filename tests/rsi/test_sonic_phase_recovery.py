"""Contact-triggered phase recovery is bounded and causal."""

import pytest

from rosclaw_soccer.rsi.sonic_phase_probe import (
    phase_offset_frames,
    recovered_phase_offset_frames,
)


def test_phase_recovery_unwinds_only_after_first_clean_foot_contact() -> None:
    assert recovered_phase_offset_frames(
        20, 6, first_foot_contact_frame=None, recovery_frames=12
    ) == phase_offset_frames(20, 6)
    assert recovered_phase_offset_frames(
        21, 6, first_foot_contact_frame=20, recovery_frames=12
    ) == pytest.approx(5.5)
    assert (
        recovered_phase_offset_frames(32, 6, first_foot_contact_frame=20, recovery_frames=12) == 0
    )
    assert (
        recovered_phase_offset_frames(50, 6, first_foot_contact_frame=20, recovery_frames=12) == 0
    )
    with pytest.raises(ValueError):
        recovered_phase_offset_frames(20, 6, first_foot_contact_frame=20, recovery_frames=12)
    with pytest.raises(ValueError):
        recovered_phase_offset_frames(21, 6, first_foot_contact_frame=20, recovery_frames=11)
