import numpy as np
import pytest

from rosclaw_soccer.rsi.pass_impulse_window import measure_first_pass_impulse


def test_waits_for_actual_impulse_end_not_first_strong_contact():
    f = np.zeros((20, 2, 3))
    f[2:7, 0, 0] = [2, 15, 30, 20, 2]
    v = np.zeros((20, 3))
    v[3, 0] = 0.2
    v[11:, 0] = 2.0
    r = measure_first_pass_impulse(f, v)
    assert r["status"] == "QUIET_EXIT_OBSERVED"
    assert (r["start"], r["first_strong"], r["last_foot_force_step"]) == (2, 3, 6)
    assert r["quiet_confirmation_step"] == 11
    assert r["first_strong_ball_speed_mps"] == 0.2
    assert r["quiet_confirmation_ball_speed_mps"] == 2.0


def test_censor_no_contact_and_interference():
    f = np.zeros((12, 2, 3))
    v = np.zeros((12, 3))
    assert measure_first_pass_impulse(f, v)["status"] == "NO_PASSER_FOOT_CONTACT"
    f[10:, 0, 0] = 11
    assert measure_first_pass_impulse(f, v)["status"] == "RIGHT_CENSORED"
    f[:] = 0
    f[2:4, 0, 0] = 11
    f[6, 1, 2] = 2
    assert measure_first_pass_impulse(f, v)["status"] == "OTHER_BODY_OR_RECEIVER_CONTACT"


def test_recontact_resets_quiet_count_and_invalid_is_rejected():
    f = np.zeros((20, 2, 3))
    f[[1, 4], 0, 0] = 12
    v = np.zeros((20, 3))
    assert measure_first_pass_impulse(f, v)["quiet_confirmation_step"] == 9
    f[0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        measure_first_pass_impulse(f, v)
