import numpy as np

from rosclaw_soccer.rsi.relay_follow_comparison import (
    equal_prefix,
    pair_metrics,
    untouched_prefix_length,
)


def test_touch_is_from_previous_physics_before_the_next_control():
    force = np.zeros((10, 10, 2, 3))
    force[3, 7, 1, 1] = 1.01
    assert untouched_prefix_length(force) == (4, True)
    assert untouched_prefix_length(np.zeros_like(force)) == (10, False)
    a, b = np.zeros((10, 2)), np.zeros((10, 2))
    b[4] = 1
    assert equal_prefix({"q": a}, {"q": b}, 4)
    assert not equal_prefix({"q": a}, {"q": b}, 5)


def test_paired_rescues_do_not_hide_regressions():
    base = dict(
        handoff=False,
        body_failure=False,
        A_failed=False,
        B_failed=False,
        outgoing_foot_strike=False,
        return_0p1m_and_healthy=False,
    )
    candidate = dict(base, handoff=True, body_failure=True, B_failed=True)
    result = pair_metrics(base, candidate)
    assert result["handoff_rescue"] and result["body_failure_regression"]
    assert not result["precise_rescue"]
