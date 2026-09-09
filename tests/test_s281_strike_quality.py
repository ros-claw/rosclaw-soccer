from dataclasses import replace

import pytest

from rosclaw_soccer.sim.strike_quality import StrikeQualityConfig, StrikeQualityTracker


def sample(tracker, index, *, foot=0.0, other=0.0, speed=4.0, forward=0.5, safe=True):
    tracker.observe(
        elapsed_sec=index * 0.002,
        body_safe=safe,
        forward_speed_mps=forward,
        directed_ball_speed_mps=speed,
        foot_normal_force_n=foot,
        other_non_ground_normal_force_n=other,
    )


def test_a_strong_strike_requires_complete_safe_forward_evidence():
    tracker = StrikeQualityTracker(StrikeQualityConfig(duration_sec=0.01))
    sample(tracker, 1)
    sample(tracker, 2, foot=2.0)
    assert not tracker.result()["strong_forward_strike"]
    for i in range(3, 6):
        sample(tracker, i)
    result = tracker.result()
    assert result["strong_forward_strike"]
    assert result["samples"] == 5
    assert result["first_foot_sec"] == 0.004
    assert result["activation_ceiling"] == "SIM_ONLY"
    assert result["promotion_eligible"] is False


@pytest.mark.parametrize("kind", ["prior_shin", "simultaneous_shin", "stopped", "retreat", "fell"])
def test_no_credit_shortcuts(kind):
    tracker = StrikeQualityTracker(StrikeQualityConfig(duration_sec=0.01))
    for i in range(1, 6):
        sample(
            tracker,
            i,
            foot=2.0 if i == 2 else 0.0,
            other=2.0
            if (kind == "prior_shin" and i == 1) or (kind == "simultaneous_shin" and i == 2)
            else 0.0,
            forward=0.0
            if kind == "stopped" and i == 1
            else (-0.5 if kind == "retreat" and i == 1 else 0.5),
            safe=not (kind == "fell" and i == 5),
        )
    assert not tracker.result()["strong_forward_strike"]


def test_run_phase_contact_is_not_strike_phase_mastery():
    tracker = StrikeQualityTracker(
        StrikeQualityConfig(duration_sec=0.01, strike_reference_start_sec=0.006)
    )
    for i in range(1, 6):
        sample(tracker, i, foot=2.0 if i == 1 else 0.0)
    assert tracker.result()["clean_touch"]
    assert not tracker.result()["reference_strike_phase_contact"]
    assert not tracker.result()["strong_forward_strike"]


def test_late_net_contact_does_not_erase_an_already_observed_foot_impulse():
    tracker = StrikeQualityTracker(StrikeQualityConfig(duration_sec=0.01))
    for i in range(1, 6):
        sample(tracker, i, foot=2.0 if i == 2 else 0.0, other=5.0 if i == 4 else 0.0)
    assert tracker.result()["strong_forward_strike"]
    assert tracker.result()["first_other_sec"] == 0.008


def test_second_kick_cannot_replace_first_contact_window():
    tracker = StrikeQualityTracker(
        StrikeQualityConfig(duration_sec=0.01, attribution_window_sec=0.002)
    )
    for i in range(1, 6):
        sample(tracker, i, foot=2.0 if i in (1, 5) else 0.0, speed=0.5 if i < 5 else 8.0)
    assert tracker.result()["clean_directed_peak_mps"] == 0.5
    assert not tracker.result()["strong_forward_strike"]


@pytest.mark.parametrize("index,value", [(2, 0.5), (1, float("nan")), (1, float("inf"))])
def test_invalid_or_missing_samples_latch_failure(index, value):
    tracker = StrikeQualityTracker(StrikeQualityConfig(duration_sec=0.01))
    with pytest.raises(ValueError):
        sample(tracker, index, forward=value)
    with pytest.raises(ValueError, match="fault-latched"):
        sample(tracker, 1)
    assert tracker.result()["faulted"]
    assert not tracker.result()["strong_forward_strike"]


def test_configuration_is_bounded_content_and_time_grid():
    original = StrikeQualityConfig()
    assert (
        original.contract_hash != replace(original, directed_ball_threshold_mps=4.0).contract_hash
    )
    for change in (
        {"duration_sec": 0.003},
        {"physics_dt_sec": 0.0},
        {"strike_reference_start_sec": 4.0},
        {"forward_speed_floor_mps": True},
    ):
        with pytest.raises(ValueError):
            replace(original, **change)
