"""Pure navigation intervention tests, not G1 performance."""

import pytest

from rosclaw_soccer.rsi.bounded_first_touch_control import BoundedFirstTouchControl
from rosclaw_soccer.rsi.constant_forward_approach import apply_constant_forward


@pytest.mark.parametrize("gap", [0.1, 0.5, 2.7])
@pytest.mark.parametrize("before", [False, True])
def test_only_before_contact_forward_changes(gap, before):
    raw, clipped = BoundedFirstTouchControl().navigation(
        gap_x=gap, gap_y=0.1, ball_vx=0.3, before_contact=before
    )
    a, b = apply_constant_forward(raw, clipped, before_contact=before, cap=1.4)
    assert a[1:] == raw[1:] and b[1:] == clipped[1:]
    assert a[0] == (1.4 if before else raw[0])
    assert b[0] == (1.4 if before else clipped[0])


@pytest.mark.parametrize("cap", [1.5, 2.0, 1, float("nan")])
def test_no_speed_cap_sweep(cap):
    with pytest.raises(ValueError):
        apply_constant_forward((0.3, 0.0, 0.0), (0.3, 0.0, 0.0), before_contact=True, cap=cap)
