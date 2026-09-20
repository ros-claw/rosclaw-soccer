from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from rosclaw_soccer.training.receiving_experiment import simulate_r0_receiving_course
from rosclaw_soccer.training.receiving_oracle_schedule import (
    ReceivingOracleCursor,
    ReceivingOracleSchedule,
)
from rosclaw_soccer.training.receiving_phase_feedback import (
    ReceivingPhaseCursor,
    ReceivingPhaseReference,
    receiving_phase_features,
)
from rosclaw_soccer.training.role_receiving_courses import ReceivingCourse


def reference(enabled=True):
    return ReceivingPhaseReference(
        "sha256:" + "1" * 64,
        "sha256:" + "2" * 64,
        2,
        tuple((float(i),) * 18 for i in range(2, 62)),
        enabled,
    )


def test_features_use_yaw_frame_and_absolute_rotated_ball_velocity():
    identity = np.r_[1.0, 2.0, 0.7, 1.0, 0.0, 0.0, 0.0]
    a = receiving_phase_features(
        identity, np.zeros(12), np.array([1.2, 2, 0.1]), np.array([1.0, 0, 0])
    )
    rotated = np.r_[4.0, 5.0, 0.7, np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)]
    b = receiving_phase_features(
        rotated, np.zeros(12), np.array([4, 5.2, 0.1]), np.array([0.0, 1, 0])
    )
    np.testing.assert_allclose(a, b, atol=1e-14)
    np.testing.assert_allclose(a[:6], [1, 0, -3, 2, 0, 0], atol=1e-14)


def test_clock_control_ignores_features_and_retains_original_schedule():
    ref = reference(False)
    phase = ReceivingPhaseCursor(ref)
    schedule = ReceivingOracleSchedule(
        "blue.playmaker", "A0_leg12", 2, 20, ((0.0,) * 12, (1.0,) * 12)
    )
    old, new = ReceivingOracleCursor(schedule), ReceivingOracleCursor(schedule)
    previous = np.zeros(12)
    for frame in range(62):
        chosen = phase.step(frame, (-100.0,) * 18)
        assert chosen == (None if frame < 2 else frame)
        left = old.step(frame, active=True, predecessor=previous)
        right = new.step(frame, active=True, predecessor=previous, reference_frame=chosen)
        np.testing.assert_array_equal(left, right)
        if left is not None:
            previous = left


def test_feedback_responds_to_current_state_with_monotonic_bounded_phase():
    fast, slow = ReceivingPhaseCursor(reference()), ReceivingPhaseCursor(reference())
    for frame in range(2):
        assert fast.step(frame, (0.0,) * 18) is None
        assert slow.step(frame, (0.0,) * 18) is None
    assert fast.step(2, (20.0,) * 18) == slow.step(2, (0.0,) * 18) == 2
    assert fast.step(3, (20.0,) * 18) == 6
    assert slow.step(3, (0.0,) * 18) == 2
    previous = 6
    for frame in range(4, 100):
        chosen = fast.step(frame, (100.0,) * 18)
        assert chosen is not None and previous <= chosen <= previous + 4
        assert chosen <= min(frame + 15, 61)
        previous = chosen


@pytest.mark.parametrize(
    "change",
    [
        {"start_frame": True},
        {"features": ()},
        {"features": ((0.0,) * 17,)},
        {"features": ((float("nan"),) * 18,)},
        {"features": ((True,) * 18,)},
        {"feedback_enabled": 1},
        {"schedule_hash": "unknown"},
        {"source_evidence_hash": "unknown"},
    ],
)
def test_invalid_reference_rejected(change):
    with pytest.raises(ValueError):
        replace(reference(), **change)


def test_fault_latches_and_instance_history_is_private():
    first, second = ReceivingPhaseCursor(reference()), ReceivingPhaseCursor(reference())
    first.step(0, (0.0,) * 18)
    assert second.next_frame == 0
    with pytest.raises(ValueError):
        first.step(0, (0.0,) * 18)
    with pytest.raises(ValueError, match="latched"):
        first.step(1, (0.0,) * 18)


@pytest.mark.parametrize("value", [True, -1, 1000, 0.5])
def test_oracle_reference_frame_cannot_expand_contract(value):
    schedule = ReceivingOracleSchedule("blue.playmaker", "A0_leg12", 0, 20, ((0.0,) * 12,))
    with pytest.raises(ValueError):
        ReceivingOracleCursor(schedule).step(
            0, active=True, predecessor=np.zeros(12), reference_frame=value
        )


def test_reference_identity_and_feedback_switch_bind_hash():
    original = reference()
    for changed in (
        replace(original, feedback_enabled=False),
        replace(original, start_frame=3),
        replace(original, source_evidence_hash="sha256:" + "3" * 64),
        replace(original, schedule_hash="sha256:" + "4" * 64),
        replace(original, features=((0.0,) * 18,)),
    ):
        assert changed.contract_hash != original.contract_hash


def test_unbound_phase_reference_rejected_before_asset_load():
    with pytest.raises(ValueError, match="bind"):
        simulate_r0_receiving_course(
            asset_root=Path("must-not-load"),
            reference_policy_path=Path("must-not-load"),
            course=ReceivingCourse("blue.playmaker", 1, 0.75, -0.08),
            scenario_id="phase.test",
            phase_reference=reference(),
        )
