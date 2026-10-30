import numpy as np
import pytest

from rosclaw_soccer.training.contact_impact_observation import first_foot_impact_observation


def test_contiguous_first_impact_and_momentum_are_separately_measured():
    v, f = np.zeros((8, 3)), np.zeros((8, 3))
    f[2:4, 0] = [100.0, 300.0]
    f[3, 2] = 5.0
    f[6, 1] = 500.0  # Separate later contact must not enter the first segment.
    v[2:, 0] = 2.0
    row = first_foot_impact_observation(v, f, timestep_sec=0.002, ball_mass_kg=0.43)
    assert row["status"] == "OBSERVED" and (row["first_substep"], row["last_substep"]) == (2, 3)
    assert row["peak_foot_force_n"] == 300 and row["duration_sec"] == 0.004
    assert row["max_channel_force_time_integral_ns"] == pytest.approx(0.8)
    assert row["ball_linear_momentum_change_kg_mps"] == pytest.approx([0.86, 0, 0])
    assert row["ball_translational_energy_change_j"] == pytest.approx(0.86)
    assert row["concurrent_nonfoot_contact"] and not row["physical_safety_proven"]
    assert not row["force_time_integral_is_ball_impulse"]


def test_missing_precontact_and_right_censored_tail_are_not_complete():
    v, f = np.zeros((4, 3)), np.zeros((4, 3))
    assert (
        first_foot_impact_observation(v, f, timestep_sec=0.002, ball_mass_kg=0.43)["status"]
        == "NO_OBSERVED_FOOT_CONTACT"
    )
    f[0, 0] = 2
    assert (
        first_foot_impact_observation(v, f, timestep_sec=0.002, ball_mass_kg=0.43)["status"]
        == "PRECONTACT_CENSORED"
    )
    f[:] = 0
    f[-1, 0] = 2
    assert (
        first_foot_impact_observation(v, f, timestep_sec=0.002, ball_mass_kg=0.43)["status"]
        == "POSTCONTACT_CENSORED"
    )


def test_invalid_data_fails_closed():
    v, f = np.zeros((4, 3)), np.zeros((4, 3))
    f[1, 0] = np.nan
    with pytest.raises(ValueError):
        first_foot_impact_observation(v, f, timestep_sec=0.002, ball_mass_kg=0.43)
    with pytest.raises(ValueError):
        first_foot_impact_observation(v, np.zeros((4, 3)), timestep_sec=0.002, ball_mass_kg=True)
