from dataclasses import replace

import numpy as np
import pytest

from rosclaw_soccer.training.receiving_oracle_schedule import (
    ReceivingOracleCursor,
    ReceivingOracleSchedule,
)
from rosclaw_soccer.training.receiving_plan import receiving_desired_plan


def schedule():
    return ReceivingOracleSchedule(
        "blue.playmaker",
        "A0_leg12",
        30,
        20,
        ((-1.0,) * 12, (0.7,) * 12, (-0.3,) * 12, (1.0,) * 12),
    )


@pytest.mark.parametrize("shift", [-100, -5, 0, 5, 100])
def test_plan_matches_original_cursor_including_filter_and_admission(shift):
    spec = schedule()
    cursor = ReceivingOracleCursor(spec)
    plan = receiving_desired_plan(
        spec, start_frame=30, control_frames=100, phase_shift_frames=shift
    )
    predecessor = np.linspace(-0.1, 0.1, 12)
    for frame in range(30):
        assert cursor.step(frame, active=True, predecessor=predecessor) is None
    old = predecessor.copy()
    for frame, row in enumerate(plan, 30):
        active = frame % 7 != 0
        desired = np.asarray(row) if active else np.zeros(12)
        old += np.clip(0.25 * (desired - old), -0.02, 0.02)
        actual = cursor.step(
            frame, active=active, predecessor=predecessor, reference_frame=max(30, frame + shift)
        )
        assert np.array_equal(actual, old)


def test_replanning_preserves_sequence_not_first_row_hold():
    spec = schedule()
    whole = receiving_desired_plan(spec, start_frame=30, control_frames=25)
    assert whole[0] != whole[1]
    for frame in range(30, 50, 5):
        tail = receiving_desired_plan(spec, start_frame=frame, control_frames=5)
        assert tail == whole[frame - 30 : frame - 25]
    assert type(whole) is tuple and all(type(row) is tuple for row in whole)
    assert max(abs(x) for row in whole for x in row) <= 0.1


@pytest.mark.parametrize(
    "kwargs",
    [
        {"start_frame": 29},
        {"start_frame": True},
        {"start_frame": 999},
        {"control_frames": 0},
        {"control_frames": 101},
        {"control_frames": True},
        {"phase_shift_frames": 101},
        {"phase_shift_frames": -101},
        {"phase_shift_frames": 1.0},
        {"phase_shift_frames": True},
    ],
)
def test_invalid_plan_rejected(kwargs):
    arguments = dict(start_frame=30, control_frames=25)
    arguments.update(kwargs)
    with pytest.raises(ValueError):
        receiving_desired_plan(schedule(), **arguments)


def test_wrong_substrate_rejected_and_single_knot_is_constant():
    spec = schedule()
    with pytest.raises(ValueError):
        receiving_desired_plan(
            replace(spec, substrate="A1_body29", knots=((0.0,) * 29,)),
            start_frame=30,
            control_frames=25,
        )
    one = replace(spec, knots=((1.0,) * 12,))
    assert receiving_desired_plan(one, start_frame=900, control_frames=100) == ((0.1,) * 12,) * 100
