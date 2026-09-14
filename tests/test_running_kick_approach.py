import math

import pytest

from rosclaw_soccer.providers.g1.running_kick_approach import (
    RunningKickApproachConfig,
    propose_running_kick_approach,
)


def propose(gap=0.8, speed=0.6, cruise=2.0, config=None):
    return propose_running_kick_approach(
        longitudinal_ball_gap_m=gap,
        measured_forward_speed_m_s=speed,
        cruising_speed_m_s=cruise,
        config=config,
    )


def test_approach_is_nonzero_monotone_capped_and_causal():
    speeds = [propose(gap=g).requested_forward_speed_m_s for g in (5.0, 2.0, 1.5, 1.0, 0.8)]
    assert speeds == sorted(speeds, reverse=True)
    assert speeds[0] == 2 and speeds[-1] == 0.5
    assert propose(gap=1.5).requested_forward_speed_m_s == math.sqrt(1.25)
    assert propose().entry_condition_met
    assert propose().activation_ceiling == "SIM_ONLY"
    assert not propose().promotion_eligible


@pytest.mark.parametrize(
    "gap,speed", [(-0.1, 0.5), (0, 0.5), (0.81, 0.5), (0.8, 0), (0.8, -1), (0.8, 0.76)]
)
def test_entry_rejects_behind_far_stationary_reverse_and_fast(gap, speed):
    assert not propose(gap, speed).entry_condition_met
    if gap <= 0:
        assert propose(gap, speed).requested_forward_speed_m_s == 0


@pytest.mark.parametrize("value", [float("nan"), float("inf"), True, "0.5"])
def test_invalid_observations_rejected(value):
    with pytest.raises(ValueError):
        propose(gap=value)
    with pytest.raises(ValueError):
        propose(speed=value)
    with pytest.raises(ValueError):
        propose(cruise=value)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(approach_deceleration_m_s2=0),
        dict(contact_speed_m_s=0),
        dict(braking_gap_m=0.7),
        dict(maximum_entry_speed_m_s=0.4),
        dict(maximum_entry_gap_m=float("nan")),
        dict(contact_speed_m_s=True),
    ],
)
def test_invalid_configs_rejected(kwargs):
    with pytest.raises(ValueError):
        RunningKickApproachConfig(**kwargs)


def test_invalid_config_and_envelope_rejected():
    for kwargs in (dict(config=False), dict(gap=31), dict(speed=11), dict(cruise=0.2)):
        with pytest.raises(ValueError):
            propose(**kwargs)
