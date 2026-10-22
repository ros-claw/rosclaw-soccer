import math

import pytest

from rosclaw_soccer.rsi.motor_exploration_profile import (
    declared_exploration_sigma,
    exploration_sigma,
)


def test_legacy_amplitude_and_new_bounded_intervention():
    assert declared_exploration_sigma(dict(sigma_policy_units=0.005)) == 0.005
    assert exploration_sigma("contact-v1", "box-truncated-ar1") == 0.02
    assert (
        declared_exploration_sigma(
            dict(
                exploration_profile="contact-v1",
                sampling_law="box-truncated-ar1",
                sigma_policy_units=0.02,
            )
        )
        == 0.02
    )


@pytest.mark.parametrize("sigma", [0.005, 0.03, math.nan, math.inf])
def test_changed_density_cannot_masquerade_as_contact_profile(sigma):
    with pytest.raises(ValueError):
        declared_exploration_sigma(
            dict(
                exploration_profile="contact-v1",
                sampling_law="box-truncated-ar1",
                sigma_policy_units=sigma,
            )
        )


@pytest.mark.parametrize(
    "profile,law", [("contact-v1", "gaussian-ar1"), ("unknown", "box-truncated-ar1")]
)
def test_unreviewed_combinations_fail_closed(profile, law):
    with pytest.raises(ValueError):
        exploration_sigma(profile, law)
