from dataclasses import FrozenInstanceError

import pytest

from rosclaw_soccer.providers.g1.kick_recovery_timing import (
    KickRecoveryTiming,
    KickRecoveryTimingConfig,
)


def timing():
    return KickRecoveryTiming(KickRecoveryTimingConfig(250, 330))


def test_measured_contact_preserves_followthrough_and_latches():
    gate = timing()
    assert gate.advance(250, first_contact_tick=None).recovery_blend == 0
    decision = gate.advance(283, first_contact_tick=2824)
    assert decision.contact_observed_frame == 283
    assert decision.recovery_start_frame == 293
    assert decision.recovery_blend == 0
    assert gate.advance(293, first_contact_tick=None).recovery_blend == 0.1
    assert gate.advance(302, first_contact_tick=2824).recovery_blend == 1
    with pytest.raises(FrozenInstanceError):
        decision.recovery_blend = 1


def test_timeout_is_bounded_even_without_contact():
    gate = timing()
    assert gate.advance(329, first_contact_tick=None).recovery_blend == 0
    assert gate.advance(330, first_contact_tick=None).recovery_blend == 0.1
    assert gate.advance(400, first_contact_tick=None).recovery_blend == 1


@pytest.mark.parametrize("tick", [True, -1, 2500.0, float("nan"), 2499, 2510])
def test_invalid_or_future_contact_does_not_mutate(tick):
    gate = timing()
    with pytest.raises(ValueError):
        gate.advance(251, first_contact_tick=tick)
    assert gate.advance(251, first_contact_tick=2500).contact_observed_frame == 251


def test_contact_cannot_be_rewritten_and_rejection_is_atomic():
    gate = timing()
    gate.advance(251, first_contact_tick=2500)
    with pytest.raises(ValueError, match="rewritten"):
        gate.advance(252, first_contact_tick=2501)
    assert gate.advance(252, first_contact_tick=None).contact_observed_frame == 251


@pytest.mark.parametrize("frame", [250, 249, True, -1, float("inf")])
def test_nonmonotonic_or_invalid_clock(frame):
    gate = timing()
    gate.advance(250, first_contact_tick=None)
    with pytest.raises(ValueError):
        gate.advance(frame, first_contact_tick=None)
    assert gate.advance(251, first_contact_tick=None).recovery_start_frame == 330


@pytest.mark.parametrize(
    "kwargs",
    [
        {"kick_start_frame": True},
        {"latest_recovery_frame": 249},
        {"followthrough_frames": -1},
        {"blend_frames": 0},
        {"physics_ticks_per_frame": 0},
        {"blend_frames": float("nan")},
        {"latest_recovery_frame": 10**10},
    ],
)
def test_config_rejects_invalid_values(kwargs):
    params = dict(kick_start_frame=250, latest_recovery_frame=330)
    params.update(kwargs)
    with pytest.raises(ValueError):
        KickRecoveryTimingConfig(**params)


def test_late_contact_cannot_delay_timeout():
    gate = timing()
    assert gate.advance(335, first_contact_tick=3340).recovery_start_frame == 330


def test_config_type_checked():
    with pytest.raises(ValueError):
        KickRecoveryTiming(None)
