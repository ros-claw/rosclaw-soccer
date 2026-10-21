import pytest

from rosclaw_soccer.rsi.contact_option_clock import enter_measured_recovery


def test_contact_control_plus14_matches_nominal_contact24_but_not_an_oracle():
    assert not enter_measured_recovery(
        completed_controls=23, first_contact_control=9, reference_horizon=260
    )
    assert enter_measured_recovery(
        completed_controls=24, first_contact_control=9, reference_horizon=260
    )
    assert not enter_measured_recovery(
        completed_controls=23, first_contact_control=None, reference_horizon=260
    )
    assert enter_measured_recovery(
        completed_controls=260, first_contact_control=None, reference_horizon=260
    )
    assert enter_measured_recovery(
        completed_controls=260, first_contact_control=259, reference_horizon=260
    )
    with pytest.raises(ValueError):
        enter_measured_recovery(
            completed_controls=10, first_contact_control=11, reference_horizon=260
        )
