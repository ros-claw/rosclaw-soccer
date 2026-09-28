"""Moving-ball contact adapter is causal and legacy-static compatible."""

import pytest

from rosclaw_soccer.sim.rolling_contact_window import contact_adapter_window_open


def _open(*, rolling: bool, seen: bool, moved: float, gap: float = 0.5) -> bool:
    return contact_adapter_window_open(
        ball_root_gap_m=gap,
        ball_origin_displacement_m=moved,
        first_contact_seen=seen,
        rolling_adapter_enabled=rolling,
        maximum_gap_m=1.0,
    )


def test_legacy_static_ball_window_unchanged() -> None:
    assert _open(rolling=False, seen=False, moved=0.04)
    assert _open(rolling=False, seen=True, moved=0.04)
    assert not _open(rolling=False, seen=False, moved=0.05)


def test_rolling_ball_uses_current_contact_latch() -> None:
    assert _open(rolling=True, seen=False, moved=0.7)
    assert not _open(rolling=True, seen=True, moved=0.7)
    assert not _open(rolling=True, seen=False, moved=0.7, gap=1.01)


def test_invalid_state_fails_closed() -> None:
    with pytest.raises(ValueError, match="finite bounded"):
        _open(rolling=True, seen=False, moved=float("nan"))
    with pytest.raises(ValueError, match="finite bounded"):
        contact_adapter_window_open(
            ball_root_gap_m=0.5,
            ball_origin_displacement_m=0.2,
            first_contact_seen=1,  # type: ignore[arg-type]
            rolling_adapter_enabled=True,
            maximum_gap_m=1.0,
        )
