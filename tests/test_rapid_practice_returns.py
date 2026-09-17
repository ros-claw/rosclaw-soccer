from dataclasses import replace

import numpy as np
import pytest
from test_training_ball_return import observe, referee

from rosclaw_soccer.training.return_turnaround import return_turnaround_metrics
from rosclaw_soccer.world.training_ball_return import TrainingBallReturnConfig


def test_rapid_preset_preserves_geometry_and_legacy_default():
    old = TrainingBallReturnConfig()
    fast = TrainingBallReturnConfig.rapid_practice()
    assert old.delay_sec == 1 and old.maximum_returns == 3
    assert fast == replace(old, delay_sec=0.5, maximum_returns=10)


def test_half_second_whole_ball_exit_delay_and_ten_return_budget():
    r = referee(delay_sec=0.5, maximum_returns=10)
    assert observe(r, 0, (0, 3.1, 0.115)) is None
    for i in range(10):
        t = 1.0 + i * 2.0
        assert observe(r, t).code == 1
        assert observe(r, t + 0.48) is None
        assert observe(r, t + 0.5).code == 2
        assert observe(r, t + 0.8, (0, 2.8, 0.7)).code == 3
    assert observe(r, 21).code == 1
    assert observe(r, 22) is None
    assert r.return_count == 10


def test_fast_return_still_waits_for_clear_release_location():
    r = referee(delay_sec=0.5, maximum_returns=10)
    observe(r, 0)
    blocked = tuple((x, y) for x in (0, 1, -0.75, 2) for y in (-3.365, 3.365))
    assert observe(r, 0.5, players=blocked) is None
    assert r.return_count == 0
    assert observe(r, 0.52).code == 2


def test_latency_does_not_hide_exhausted_budget_or_failed_throw():
    t = np.array([0, 1, 1.5, 1.76, 4, 4.5, 8, 9], dtype=float)
    codes = np.array([0, 1, 2, 3, 1, 2, 1, 0])
    r = return_turnaround_metrics(t, codes)
    assert r["external_releases"] == 2 and r["observed_reentries"] == 1
    assert r["unreleased_exits"] == 1
    assert r["rows"][0]["exit_to_release_sec"] == 0.5
    assert r["rows"][1]["exit_to_reentry_sec"] is None
    assert r["rows"][2]["exit_to_release_sec"] is None


def test_latency_rejects_fabricated_reentry_and_stale_time():
    with pytest.raises(ValueError):
        return_turnaround_metrics(np.array([0.0, 1.0]), np.array([0, 3]))
    with pytest.raises(ValueError):
        return_turnaround_metrics(np.array([0.0, 0.0]), np.array([0, 1]))
