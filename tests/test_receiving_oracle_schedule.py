from dataclasses import replace

import numpy as np
import pytest

from rosclaw_soccer.training.receiving_oracle_schedule import (
    ReceivingOracleCursor,
    ReceivingOracleSchedule,
)


def schedule(kind="A0_leg12", start=0):
    dim = 12 if kind == "A0_leg12" else 29
    return ReceivingOracleSchedule("blue.playmaker", kind, start, 10, ((1.0,) * dim,))


@pytest.mark.parametrize("kind", ["A0_leg12", "A1_body29", "A3_sonic_residual"])
def test_bounded_private_filters_and_returned_arrays(kind):
    cursor = ReceivingOracleCursor(schedule(kind))
    previous = np.zeros(12)
    last = np.zeros(len(cursor.schedule.knots[0]))
    for frame in range(200):
        value = cursor.step(frame, active=frame < 100, predecessor=previous)
        assert value is not None
        assert np.max(abs(value)) <= 0.1
        assert np.max(abs(value - last)) <= 0.020000001
        last = value.copy()
        previous = value[:12].copy()
        value[:] = 20  # caller mutation must not corrupt private history
    assert np.max(abs(last)) < 1e-10


def test_common_prefix_and_measured_predecessor():
    cursor = ReceivingOracleCursor(schedule("A1_body29", start=2))
    old = np.full(12, -0.1)
    assert cursor.step(0, active=True, predecessor=old) is None
    assert cursor.step(1, active=True, predecessor=old) is None
    value = cursor.step(2, active=True, predecessor=old)
    np.testing.assert_allclose(value[:12], -0.08)
    np.testing.assert_allclose(value[12:], 0.02)


def test_interpolation_and_terminal_knot_are_declared():
    spec = replace(schedule(), knots=((0.0,) * 12, (1.0,) * 12))
    cursor = ReceivingOracleCursor(spec)
    old = np.zeros(12)
    for frame in range(30):
        desired = min(frame / 10, 1) * 0.1
        expected = old + np.clip(0.25 * (desired - old), -0.02, 0.02)
        value = cursor.step(frame, active=True, predecessor=old)
        np.testing.assert_allclose(value, expected)
        old = value


@pytest.mark.parametrize(
    "change",
    [
        {"substrate": "A2_sonic"},
        {"start_frame": True},
        {"knot_frames": 0},
        {"knots": ()},
        {"knots": ((float("nan"),) * 12,)},
        {"knots": ((True,) * 12,)},
        {"knots": ((1.1,) * 12,)},
        {"knots": ((0.0,) * 29,)},
    ],
)
def test_bad_schedule_rejected(change):
    with pytest.raises(ValueError):
        replace(schedule(), **change)


def test_fault_latches_and_separate_rollouts_do_not_share_memory():
    one, two = ReceivingOracleCursor(schedule()), ReceivingOracleCursor(schedule())
    one.step(0, active=True, predecessor=np.zeros(12))
    assert two.next_frame == 0 and two.previous is None
    with pytest.raises(ValueError):
        one.step(0, active=True, predecessor=np.zeros(12))
    with pytest.raises(ValueError, match="latched"):
        one.step(1, active=True, predecessor=np.zeros(12))


def test_all_schedule_fields_bind_hash():
    original = schedule()
    for changed in (
        replace(original, start_frame=1),
        replace(original, agent_id="red.defender"),
        replace(original, knots=((0.0,) * 12,)),
        schedule("A1_body29"),
    ):
        assert changed.contract_hash != original.contract_hash
