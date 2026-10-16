import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from rosclaw_soccer.rsi.bounded_first_touch_control import BoundedFirstTouchControl
from rosclaw_soccer.rsi.native_first_touch_episode import run_native_first_touch_episode
from rosclaw_soccer.rsi.postcontact_navigation_probe import PostcontactNavigationProbe


def test_near_slow_ball_does_not_trigger_full_speed_chase():
    p = PostcontactNavigationProbe()
    raw, command = p.navigation(previous_forward=0.4, gap_x=0.1, ball_vx=0.1, forward_cap=1.4)
    assert raw == pytest.approx(-0.35)
    assert command == pytest.approx(0.37)
    assert p.contract()["reference_slew_is_not_physical_acceleration"] is True
    assert p.contract()["trained_policy"] is False


def test_same_gap_and_ball_speed_can_accelerate_within_existing_cap():
    p = PostcontactNavigationProbe()
    raw, command = p.navigation(previous_forward=1.39, gap_x=0.55, ball_vx=3.0, forward_cap=1.4)
    assert raw == 3.0
    assert command == 1.4


def test_random_sequences_bound_reference_without_reversing_or_expanding_envelope():
    rng = np.random.default_rng(20261016)
    p = PostcontactNavigationProbe()
    prior = 1.4
    for _ in range(1000):
        x, vx = (float(v) for v in rng.uniform(-3, 3, size=2))
        _, actual = p.navigation(previous_forward=prior, gap_x=x, ball_vx=vx, forward_cap=1.4)
        assert 0.0 <= actual <= 1.4
        assert abs(actual - prior) <= 0.03 + 1e-15
        prior = actual


@pytest.mark.parametrize("key", ["target_gap_m", "separation_gain_s_inv", "reference_slew_m_s2"])
@pytest.mark.parametrize("value", [True, 0.0, float("nan"), float("inf"), 100.0])
def test_invalid_parameters_reject(key, value):
    with pytest.raises(ValueError):
        replace(PostcontactNavigationProbe(), **{key: value})


@pytest.mark.parametrize("value", [True, -0.01, 1.41, float("nan")])
def test_invalid_previous_command_is_not_silently_repaired(value):
    with pytest.raises(ValueError):
        PostcontactNavigationProbe().navigation(
            previous_forward=value, gap_x=0.5, ball_vx=0.0, forward_cap=1.4
        )


def test_unbound_probe_refuses_before_allocating_any_models():
    with pytest.raises(ValueError, match="postcontact"):
        run_native_first_touch_episode(
            argparse.Namespace(),
            course=(2.5, 0.0, -0.5),
            partition="DECLARED_DEVELOPMENT_DIAGNOSTIC",
            entry_source_path=Path(__file__),
            consumed_bank_hash=None,
            postcontact_profile=PostcontactNavigationProbe(),
        )


def test_wrong_type_refuses_before_allocation():
    with pytest.raises(ValueError, match="typed SIM"):
        run_native_first_touch_episode(
            argparse.Namespace(),
            course=(2.5, 0.0, -0.5),
            partition="DECLARED_DEVELOPMENT_DIAGNOSTIC",
            entry_source_path=Path(__file__),
            consumed_bank_hash=None,
            control_profile=BoundedFirstTouchControl(),
            approach_mode="bounded_approach_diagnostic",
            recurrent_sampling_factory=object(),
            postcontact_profile={},
        )
