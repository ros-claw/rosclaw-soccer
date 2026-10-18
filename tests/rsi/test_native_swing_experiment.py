"""Direction eligibility parity; no simulated success claimed by unit tests."""

import argparse
from pathlib import Path

import pytest

from rosclaw_soccer.rsi.native_first_touch_episode import run_native_first_touch_episode
from rosclaw_soccer.rsi.native_swing_experiment import native_swing_eligible


@pytest.mark.parametrize("memory_gate", [True, False])
@pytest.mark.parametrize("foundation_only", [True, False])
@pytest.mark.parametrize("vx", [-0.7, -1e-9, 0.0, 1e-9, 0.7])
def test_legacy_parity_and_single_condition_intervention(memory_gate, foundation_only, vx):
    args = dict(memory_gate=memory_gate, initial_vx=vx, foundation_only=foundation_only)
    assert native_swing_eligible(**args) == (memory_gate and vx < 0 and not foundation_only)
    assert native_swing_eligible(**args, mode="memory_gate_both_directions_diagnostic") == (
        memory_gate and not foundation_only
    )
    assert native_swing_eligible(**args, mode="bounded_contact_diagnostic") is (not foundation_only)


@pytest.mark.parametrize("bad", [None, True, [], "", "FRESH"])
def test_bad_mode_rejected_before_model_allocation(bad):
    with pytest.raises(ValueError, match="swing eligibility"):
        run_native_first_touch_episode(
            argparse.Namespace(),
            course=(2.2, 0.1, 0.4),
            partition="DECLARED_DEVELOPMENT_DIAGNOSTIC",
            entry_source_path=Path(__file__),
            consumed_bank_hash=None,
            swing_mode=bad,
        )


@pytest.mark.parametrize(
    "key,value",
    [
        ("memory_gate", 1),
        ("foundation_only", 0),
        ("initial_vx", True),
        ("initial_vx", 0),
        ("initial_vx", float("nan")),
        ("initial_vx", float("inf")),
    ],
)
def test_bad_inputs(key, value):
    args = dict(memory_gate=True, initial_vx=0.4, foundation_only=False)
    args[key] = value
    with pytest.raises(ValueError):
        native_swing_eligible(**args)
