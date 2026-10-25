import numpy as np
import pytest

from rosclaw_soccer.rsi.native_contact_authority_metrics import contact_authority_metrics


def arrays(n=300):
    return dict(
        commands=np.zeros((n, 8, 3)),
        ball_forces=np.zeros((n * 10, 8, 3)),
        nearest_foot_relative_speed_mps=np.full((n, 8), 0.1),
        foot_gap_after_m=np.full((n, 8), 0.2),
        teacher_ball_position_after=np.stack(
            (np.arange(n) * 0.1, np.zeros(n), np.zeros(n)), axis=1
        ),
        teacher_applied_delta=np.zeros((n, 29)),
        teacher_foot_velocity=np.zeros((n, 3)),
        teacher_ball_velocity=np.zeros((n, 3)),
        teacher_relative_velocity_after=np.zeros((n, 3)),
        teacher_active=np.zeros(n, np.bool_),
        teacher_fallback=np.zeros(n, np.bool_),
    )


@pytest.mark.parametrize("frame,observed", [(239, True), (240, False), (260, False), (299, False)])
def test_diagnostic_contact_followup_has_no_fabricated_censored_tail(frame, observed):
    data = arrays()
    data["ball_forces"][frame * 10 : frame * 10 + 7, 3, 0] = 5
    result = contact_authority_metrics(data)
    # Seven native2ms contact samples are only one20ms contact control.
    assert result["actual_foot_contact_controls"] == 1
    assert result["first_postwarmup_foot_contact_control"] == frame
    assert result["diagnostic_followup60_observed"] is observed
    assert result["diagnostic_followup_status"] == ("OBSERVED" if observed else "RIGHT_CENSORED")
    if observed:
        np.testing.assert_allclose(result["diagnostic_ball_displacement60_m"], [6, 0, 0])
    else:
        assert result["diagnostic_ball_displacement60_m"] is None
    assert not result["official_exam_pass_proven"]


def test_no_contact_and_initial_prefix_are_not_qualified_possession():
    data = arrays()
    data["ball_forces"][1, 3, 2] = 3
    result = contact_authority_metrics(data)
    assert result["initial_contact_prefix"]
    assert result["diagnostic_followup_status"] == "NO_POSTWARMUP_FOOT_CONTACT"
    assert result["diagnostic_ball_displacement60_m"] is None
    assert result["close_recent_comotion_controls"] == 0
    assert not result["physical_possession_qualified"]


def test_recent_comotion_expires_and_slew_diagnostic_uses_actual_corrections():
    data = arrays()
    data["ball_forces"][60 * 10, 3, 1] = 2
    data["teacher_applied_delta"][60:100, 0] = 0.01
    result = contact_authority_metrics(data)
    assert result["close_recent_comotion_controls"] == 26
    assert result["teacher_nonzero_correction_controls"] == 40
    assert result["maximum_nonfallback_correction_slew_rad"] == 0.01


def test_nonfinite_measurements_fail_closed():
    data = arrays()
    data["teacher_applied_delta"][50, 0] = np.nan
    with pytest.raises(ValueError):
        contact_authority_metrics(data)
