import pytest

from rosclaw_soccer.providers.g1.sonic_command_scale import SonicCommandScaleSchedule
from rosclaw_soccer.sim.contracts import hash_json


def test_attenuation_interpolation_and_final_hold():
    schedule = SonicCommandScaleSchedule((1.0, 0.0), knot_frames=20)
    assert [schedule.at(i) for i in (0, 10, 20, 1000)] == [1, 0.5, 0, 0]
    assert schedule.contract_hash != SonicCommandScaleSchedule((0.5, 0.0)).contract_hash


@pytest.mark.parametrize("knots", [(), (True,), (-0.1,), (1.01,), (float("nan"),)])
def test_reject_unsafe_scale(knots):
    with pytest.raises(ValueError):
        SonicCommandScaleSchedule(knots)


def test_reject_nonlocal_frame():
    for value in (-1, True, 0.5):
        with pytest.raises(ValueError):
            SonicCommandScaleSchedule((1.0,)).at(value)


def test_default_hash_remains_legacy(monkeypatch):
    from test_receiving_sonic import motor

    option = motor(monkeypatch)
    expected = hash_json(
        {
            "schema": "soccer.receiving_sonic_option.v1",
            "navigation": option.navigation.contract_hash,
            "start_frame": 2,
            "velocity_scale": 1.0,
            "history": "cold_start_at_measured_entry_not_hidden_state_transfer",
            "activation_ceiling": "SIM_ONLY",
        }
    )
    assert option.contract_hash == expected


def test_private_delayed_scale_cannot_enlarge_command(monkeypatch):
    from test_receiving_sonic import motor
    from test_s304_sonic_navigation import observation

    option = motor(
        monkeypatch,
        velocity_scale=0.5,
        command_scale_schedule=SonicCommandScaleSchedule((1.0, 0.0), knot_frames=2),
    )
    for frame in range(5):
        option.propose(observation(frame))
    assert [row.navigation_command[0] for row in option.navigation.calls] == [0.25, 0.125, 0.0]
    assert option.command_scale_records == [(0, 0.5), (1, 0.25), (2, 0.0)]
    assert len(option.navigation.starts) == 1


@pytest.mark.parametrize("branch", [0, 1, 19, 20, 21, 39, 40, 60])
def test_future_attenuation_preserves_exact_interpolated_past(branch):
    from rosclaw_soccer.providers.g1.sonic_command_scale import future_attenuations

    schedule = SonicCommandScaleSchedule((1.0, 0.7, 0.9, 0.8, 0.6))
    proposals = future_attenuations(schedule, local_branch_frame=branch)
    assert proposals[0] is schedule
    assert len(proposals) == 5
    for proposal in proposals:
        assert [proposal.at(t) for t in range(branch)] == [schedule.at(t) for t in range(branch)]
        assert all(proposal.at(t) <= schedule.at(t) for t in range(branch, 150))


@pytest.mark.parametrize("branch", [True, -1, 1.5, 1001, 81])
def test_future_attenuation_rejects_invalid_or_exhausted_prefix(branch):
    from rosclaw_soccer.providers.g1.sonic_command_scale import future_attenuations

    with pytest.raises(ValueError):
        future_attenuations(SonicCommandScaleSchedule((1.0,) * 5), local_branch_frame=branch)
