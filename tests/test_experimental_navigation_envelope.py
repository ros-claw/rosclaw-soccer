from dataclasses import replace

import numpy as np
import pytest
from test_s304_sonic_navigation import controller, observation

from rosclaw_soccer.providers.g1.sonic_navigation import SonicNavigationConfig
from rosclaw_soccer.skills.team.independent_team_world import IndependentTeamWorldConfig
from rosclaw_soccer.skills.team.navigation_envelope import SimulationNavigationEnvelope
from rosclaw_soccer.world.player_clearance import propose_clearance_velocity


def envelope(**changes):
    return SimulationNavigationEnvelope(
        **dict(
            agent_id="red.defender",
            motor_contract_hash="sha256:" + "a" * 64,
            maximum_speed_mps=1.5,
        )
        | changes
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"maximum_speed_mps": True},
        {"maximum_speed_mps": 0.7},
        {"maximum_speed_mps": 1.50001},
        {"maximum_speed_mps": float("nan")},
        {"maximum_speed_mps": float("inf")},
        {"agent_id": "unbound player"},
        {"motor_contract_hash": "unbound"},
        {"activation_ceiling": "REAL"},
    ],
)
def test_envelope_is_explicit_bounded_and_simulation_only(changes):
    with pytest.raises(ValueError):
        envelope(**changes)


def test_old_observations_cannot_implicitly_expand_speed():
    with pytest.raises(ValueError):
        replace(observation(), navigation_command=(1.2, 0.0, 0.0))
    fast = replace(
        observation(), navigation_command=(1.2, 0.0, 0.0), navigation_envelope=envelope()
    )
    assert fast.navigation_command[0] == 1.2
    for changes in (
        {"navigation_command": (1.4, 1.4, 0.0)},
        {"navigation_command": (1.0, 0.0, 1.6)},
        {"navigation_command": None},
        {"navigation_envelope": envelope(agent_id="blue.defender")},
        {"navigation_envelope": {}},
    ):
        with pytest.raises(ValueError):
            replace(fast, **changes)


def test_only_matching_private_sonic_accepts_expanded_command(monkeypatch):
    motor = controller(monkeypatch, SonicNavigationConfig(experimental_maximum_speed_mps=1.5))
    first = replace(
        observation(),
        navigation_command=(1.2, 0.0, 0.0),
        navigation_envelope=motor.navigation_envelope,
    )
    motor.propose(first)
    assert motor.backend.command == (1.2, 0.0, 0.0)
    with pytest.raises(ValueError, match="latched"):
        motor.propose(replace(first, frame=1, time_sec=0.02, navigation_envelope=envelope()))
    with pytest.raises(ValueError, match="latched"):
        motor.propose(replace(first, frame=1, time_sec=0.02))


def test_old_sonic_does_not_inherit_another_motor_envelope(monkeypatch):
    motor = controller(monkeypatch)
    assert motor.navigation_envelope is None
    with pytest.raises(ValueError, match="latched"):
        motor.propose(replace(observation(), navigation_envelope=envelope()))


def test_experimental_world_requires_clearance_and_distinct_sorted_players():
    with pytest.raises(ValueError):
        IndependentTeamWorldConfig(experimental_navigation_envelopes=(envelope(),))
    good = IndependentTeamWorldConfig(
        all_role_clearance=True,
        predictive_separation=True,
        experimental_navigation_envelopes=(envelope(),),
    )
    assert good.config_hash != replace(good, experimental_navigation_envelopes=()).config_hash
    for invalid in ([], (envelope(), envelope()), (object(),)):
        with pytest.raises(ValueError):
            replace(good, experimental_navigation_envelopes=invalid)


def test_fast_clearance_is_opt_in_and_does_not_rescale_constrained_command():
    nominal, neighbors = np.array([1.2, 0.3]), np.array([[0.8, 0.0]])
    with pytest.raises(ValueError):
        propose_clearance_velocity(nominal, neighbors, maximum_speed_mps=1.5)
    proposal = propose_clearance_velocity(
        nominal, neighbors, maximum_speed_mps=1.5, experimental_fast_navigation=True
    )
    assert proposal.feasible and proposal.constrained
    assert abs(proposal.velocity_mps[0]) < 1e-10
    assert abs(proposal.velocity_mps[1] - 0.3) < 1e-10
    free = propose_clearance_velocity(
        nominal, np.empty((0, 2)), maximum_speed_mps=1.5, experimental_fast_navigation=True
    )
    assert free.velocity_mps == (1.2, 0.3)
