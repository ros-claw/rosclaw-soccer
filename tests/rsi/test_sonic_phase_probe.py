"""SIM_ONLY reference clock limits and causal onset."""

import pytest

from rosclaw_soccer.rsi.sonic_phase_probe import phase_offset_frames


def test_phase_probe_ramps_without_snapshot_jump() -> None:
    assert phase_offset_frames(0, 6) == 0
    assert phase_offset_frames(10, 6) == 3
    assert phase_offset_frames(20, 6) == 6
    assert phase_offset_frames(40, -6) == -6
    assert phase_offset_frames(20, 0) == 0


@pytest.mark.parametrize("target", [6.1, float("nan"), float("inf"), True])
def test_phase_probe_rejects_out_of_bounds(target: float) -> None:
    with pytest.raises(ValueError):
        phase_offset_frames(0, target)


def test_phase_probe_rejects_too_fast_slew() -> None:
    with pytest.raises(ValueError):
        phase_offset_frames(1, 6, ramp_frames=11)
