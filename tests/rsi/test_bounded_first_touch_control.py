import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from rosclaw_soccer.rsi.bounded_first_touch_control import BOUNDS, BoundedFirstTouchControl
from rosclaw_soccer.rsi.native_first_touch_episode import run_native_first_touch_episode
from rosclaw_soccer.rsi.native_swing_experiment import native_swing_eligible


def test_commands_keep_requests_and_project_without_exceeding_envelope():
    policy = BoundedFirstTouchControl(forward_cap_m_s=1.48)
    raw, actual = policy.navigation(gap_x=3.0, gap_y=1.0, ball_vx=0.7, before_contact=True)
    assert raw[0] > actual[0]
    assert actual == (1.48, 0.2, 0.0)
    assert np.hypot(*actual[:2]) < 1.5
    _, close = policy.navigation(gap_x=0.35, gap_y=0.1, ball_vx=0.4, before_contact=True)
    assert close == (0.4, 0.0, 0.0)
    _, after = policy.navigation(gap_x=0.35, gap_y=0.1, ball_vx=0.4, before_contact=False)
    assert after == (1.48, 0.0, 0.0)
    assert policy.contract()["trained_policy"] is False


@pytest.mark.parametrize("name", list(BOUNDS))
def test_parameter_limits_cannot_be_bypassed(name):
    with pytest.raises(ValueError):
        replace(BoundedFirstTouchControl(), **{name: BOUNDS[name][1] + 1.0})
    with pytest.raises(ValueError):
        replace(BoundedFirstTouchControl(), **{name: float("nan")})


def test_new_contact_permission_is_explicit_and_foundation_only_still_excludes_it():
    assert native_swing_eligible(
        memory_gate=False, initial_vx=0.5, foundation_only=False, mode="bounded_contact_diagnostic"
    )
    assert not native_swing_eligible(
        memory_gate=True, initial_vx=0.5, foundation_only=True, mode="bounded_contact_diagnostic"
    )


@pytest.mark.parametrize(
    "partition", ["DECLARED_PUBLIC_TRAINING", "CONSUMED_TRANSFER_DIAGNOSTIC", "FRESH"]
)
def test_search_interface_never_leaks_into_legacy_training_or_exam(partition):
    with pytest.raises(ValueError):
        run_native_first_touch_episode(
            argparse.Namespace(),
            course=(2.3, 0.1, 0.5),
            partition=partition,
            entry_source_path=Path(__file__),
            consumed_bank_hash=None,
            approach_mode="bounded_approach_diagnostic",
            control_profile=BoundedFirstTouchControl(),
            recurrent_sampling_factory=object(),
        )


def test_unbound_new_modes_refuse_before_allocation():
    with pytest.raises(ValueError, match="profile"):
        run_native_first_touch_episode(
            argparse.Namespace(),
            course=(2.3, 0.1, 0.5),
            partition="DECLARED_DEVELOPMENT_DIAGNOSTIC",
            entry_source_path=Path(__file__),
            consumed_bank_hash=None,
            approach_mode="bounded_approach_diagnostic",
        )
