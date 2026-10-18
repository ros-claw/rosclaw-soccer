"""Values-only planner authority contract, not a native body or learning test."""

import struct

import pytest

from rosclaw_soccer.rsi.planner_navigation_pulse import PlannerNavigationPulse


def test_zero_all300_and_active_pulse_outside_window_are_bit_preserving():
    base = (1.4, -0.0, 0.0)
    expected = struct.pack("ddd", *base)
    for pulse in (PlannerNavigationPulse(), PlannerNavigationPulse(-0.08)):
        for frame in range(300):
            request, command = pulse.compose(frame, base)
            if pulse.lateral_delta_m_s == 0 or not 100 <= frame < 160:
                assert struct.pack("ddd", *command) == expected
                assert command is base and request == (0.0, 0.0, 0.0)
            else:
                assert request == (0.0, -0.08, 0.0)
                assert command == (1.4, -0.08, 0.0)


def test_fixed_planner_events_hold_and_reset_without_any_torque_authority():
    pulse = PlannerNavigationPulse(-0.08)
    events = [f for f in range(0, 300, 20) if pulse.compose(f, (1.4, 0.1, 0.0))[0][1]]
    assert events == [100, 120, 140]
    assert pulse.compose(160, (1.4, 0.1, 0.0))[1] == (1.4, 0.1, 0.0)
    c = pulse.contract()
    assert c["reference_and_encoder_onset_must_be_measured"]
    assert not c["trained_policy"] and not c["runtime_execution_authorized"]
    assert not c["physics_qualified"] and not c["training_authorized"]
    assert pulse.compose(100, (1.48, -0.18, 0.0))[1] == (1.48, -0.2, 0.0)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(lateral_delta_m_s=-0.1),
        dict(lateral_delta_m_s=True),
        dict(lateral_delta_m_s=0),
        dict(start_frame=80),
        dict(end_frame=180),
        dict(end_frame=True),
    ],
)
def test_does_not_open_a_new_unbounded_pulse_parameter_search(kwargs):
    with pytest.raises(ValueError):
        PlannerNavigationPulse(**kwargs)


@pytest.mark.parametrize(
    "frame,baseline",
    [
        (-1, (1.4, 0.0, 0.0)),
        (300, (1.4, 0.0, 0.0)),
        (True, (1.4, 0.0, 0.0)),
        (100, (2.0, 0.0, 0.0)),
        (100, (1.4, 0.3, 0.0)),
        (100, (1.4, 0.0, 0.1)),
        (100, (1.4, float("nan"), 0.0)),
        (100, [1.4, 0.0, 0.0]),
    ],
)
def test_only_existing_nonturning_command_envelope_and_original_clock(frame, baseline):
    with pytest.raises(ValueError):
        PlannerNavigationPulse(-0.08).compose(frame, baseline)
