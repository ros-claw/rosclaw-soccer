"""Command-law parity and opt-in scope, not improved physical outcomes."""

import argparse
from pathlib import Path

import numpy as np
import pytest

from rosclaw_soccer.rsi.native_first_touch_episode import run_native_first_touch_episode
from rosclaw_soccer.rsi.native_lateral_experiment import native_lateral_command


@pytest.mark.parametrize("tracking", [True, False])
@pytest.mark.parametrize("before_contact", [True, False])
def test_legacy_exact_original_formula(tracking, before_contact):
    rng = np.random.default_rng(1095)
    for gap_x, gap_y in rng.uniform(-3, 3, (300, 2)):
        old = (
            float(np.clip(1.2 * gap_y, -0.2, 0.2))
            if tracking and gap_x > 0.95 and before_contact
            else 0.0
        )
        assert (
            native_lateral_command(
                gap_x=float(gap_x),
                gap_y=float(gap_y),
                tracking=tracking,
                before_contact=before_contact,
            )
            == old
        )


def test_intervention_opens_both_sides_and_close_window_only_until_contact():
    for tracking in (True, False):
        for gap_y in (-1.0, 1.0):
            value = native_lateral_command(
                gap_x=0.2,
                gap_y=gap_y,
                tracking=tracking,
                before_contact=True,
                mode="continuous_lateral_diagnostic",
            )
            assert value == np.copysign(0.2, gap_y)
            assert (
                native_lateral_command(
                    gap_x=0.2,
                    gap_y=gap_y,
                    tracking=tracking,
                    before_contact=False,
                    mode="continuous_lateral_diagnostic",
                )
                == 0.0
            )


@pytest.mark.parametrize("bad", [None, True, "FRESH", "", []])
def test_invalid_mode_rejects_before_physics_allocation(bad, tmp_path):
    with pytest.raises(ValueError, match="diagnostic mode"):
        run_native_first_touch_episode(
            argparse.Namespace(),
            course=(2.2, 0.1, -0.4),
            partition="DECLARED_DEVELOPMENT_DIAGNOSTIC",
            entry_source_path=Path(__file__),
            consumed_bank_hash=None,
            approach_mode=bad,
        )
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "key,value",
    [
        ("gap_x", np.nan),
        ("gap_y", np.inf),
        ("gap_x", True),
        ("gap_y", 0),
        ("tracking", 1),
        ("before_contact", None),
    ],
)
def test_invalid_command_inputs(key, value):
    args = dict(gap_x=0.2, gap_y=0.1, tracking=True, before_contact=True)
    args[key] = value
    with pytest.raises(ValueError):
        native_lateral_command(**args)
