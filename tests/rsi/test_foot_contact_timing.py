import numpy as np
import pytest

from rosclaw_soccer.rsi.foot_contact_timing import contact_timing_metrics


def test_post_step_times_and_wrong_foot_precedence_are_explicit():
    force = np.array([[0.0, 0.0], [0.0, 2.0], [3.0, 0.0], [0.0, 0.0]])
    speed = np.array([0.0, 2.0, 1.0, 3.0])
    result = contact_timing_metrics(force, speed, intended_foot=0, timestep=0.002)
    assert result["first_foot_contact_side"] == "right"
    assert result["first_right_contact_time_s"] == 0.004
    assert result["first_left_contact_time_s"] == 0.006
    assert result["other_foot_contact_before_intended"]
    assert result["maximum_ball_speed_after_intended_contact_mps"] == 3.0
    assert not result["contact_chronology_proves_causal_kick"]


@pytest.mark.parametrize("mode", ["none", "left", "right", "simultaneous"])
def test_no_contact_threshold_and_simultaneous_contacts(mode):
    force = np.ones((4, 2))
    if mode in ("left", "simultaneous"):
        force[1, 0] = 2
    if mode in ("right", "simultaneous"):
        force[1, 1] = 2
    result = contact_timing_metrics(force, np.zeros(4), intended_foot=0, timestep=0.002)
    assert result["first_foot_contact_side"] == mode
    assert result["other_foot_contact_before_intended"] == (mode == "right")
    assert (result["maximum_ball_speed_after_intended_contact_mps"] is None) == (
        mode in ("none", "right")
    )


@pytest.mark.parametrize("bad", ["shape", "nan", "negative", "side", "clock", "integer"])
def test_invalid_aligned_evidence_rejected(bad):
    force = np.zeros((4, 2))
    speed = np.zeros(4)
    side, dt = 0, 0.002
    if bad == "shape":
        speed = np.zeros(3)
    elif bad == "nan":
        force[0, 0] = np.nan
    elif bad == "negative":
        speed[0] = -1
    elif bad == "side":
        side = True
    elif bad == "clock":
        dt = float("inf")
    else:
        force = force.astype(int)
    with pytest.raises(ValueError):
        contact_timing_metrics(force, speed, intended_foot=side, timestep=dt)
