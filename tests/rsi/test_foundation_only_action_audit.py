import argparse
from pathlib import Path

import numpy as np
import pytest

from rosclaw_soccer.rsi.bounded_first_touch_control import BoundedFirstTouchControl
from rosclaw_soccer.rsi.foundation_only_action_audit import require_foundation_only_actions
from rosclaw_soccer.rsi.native_first_touch_episode import run_native_first_touch_episode


def fixture():
    return {
        key: np.zeros((300, 1, width))
        for key, width in (
            ("foundation_neural_joint_target_mujoco_rad", 29),
            ("foundation_navigation_joint_target_rad", 29),
            ("pre_motor_joint_target_rad", 29),
            ("joint_target_rad", 29),
            ("taskspace_joint_increment_rad", 29),
            ("motor_delta_rad", 12),
            ("motor_pre_projection_requested_rad", 12),
        )
    }


def test_pure_foundation_chain_accepts_without_modifying_input():
    trace = fixture()
    require_foundation_only_actions(trace)
    assert all(not value.any() for value in trace.values())


@pytest.mark.parametrize("key", list(fixture()))
@pytest.mark.parametrize("mutation", ["nan", "shape", "missing", "increment"])
def test_corrupt_or_nonzero_chain_rejected(key, mutation):
    trace = fixture()
    if mutation == "nan":
        trace[key][123, 0, 0] = np.nan
    elif mutation == "shape":
        trace[key] = trace[key][:-1]
    elif mutation == "missing":
        del trace[key]
    else:
        trace[key][123, 0, 0] = 0.01
    with pytest.raises(ValueError):
        require_foundation_only_actions(trace)


@pytest.mark.parametrize(
    "change",
    [
        dict(foundation_approach_probe=1),
        dict(foundation_only=False),
        dict(record_foundation_observation=False),
        dict(control_profile=None),
        dict(approach_mode="legacy"),
        dict(swing_mode="bounded_contact_diagnostic"),
        dict(partition="DECLARED_PUBLIC_TRAINING"),
    ],
)
def test_invalid_ablation_rejected_before_model_allocation(change, tmp_path):
    parameters = dict(
        foundation_approach_probe=True,
        control_profile=BoundedFirstTouchControl(),
        approach_mode="bounded_approach_diagnostic",
        swing_mode="legacy",
        partition="DECLARED_DEVELOPMENT_DIAGNOSTIC",
    )
    args = dict(foundation_only=True, record_foundation_observation=True)
    for key, value in change.items():
        (args if key in args else parameters)[key] = value
    with pytest.raises(ValueError):
        run_native_first_touch_episode(
            argparse.Namespace(**args),
            course=(2.5, 0.0, -0.5),
            entry_source_path=Path(__file__),
            consumed_bank_hash=None,
            **parameters,
        )
    assert not list(tmp_path.iterdir())


def test_valid_ablation_reaches_allocator_without_factory(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace

    def unopened(path):
        raise RuntimeError("allocator boundary reached, no physics performed")

    monkeypatch.setitem(
        sys.modules, "mujoco", SimpleNamespace(MjSpec=SimpleNamespace(from_file=unopened))
    )
    args = argparse.Namespace(
        foundation_only=True, record_foundation_observation=True, scene=tmp_path / "missing.xml"
    )
    with pytest.raises(RuntimeError, match="allocator boundary reached"):
        run_native_first_touch_episode(
            args,
            course=(2.5, 0.0, -0.5),
            partition="DECLARED_DEVELOPMENT_DIAGNOSTIC",
            entry_source_path=Path(__file__),
            consumed_bank_hash=None,
            foundation_approach_probe=True,
            control_profile=BoundedFirstTouchControl(),
            approach_mode="bounded_approach_diagnostic",
        )
