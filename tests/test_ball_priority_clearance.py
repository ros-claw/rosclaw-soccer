from dataclasses import replace

import numpy as np
import pytest

from rosclaw_soccer.training.continuous_competitive_match_growth import (
    default_continuous_match_config,
)
from rosclaw_soccer.world.player_clearance import (
    ball_priority_clearance_m,
    propose_clearance_velocity,
)


def test_only_explicit_ball_participant_gets_bounded_reduction():
    assert (
        ball_priority_clearance_m(
            agent_id="red.defender", priority_agent_id="red.defender", enabled=True
        )
        == 0.6
    )
    assert (
        ball_priority_clearance_m(
            agent_id="blue.finisher", priority_agent_id="red.defender", enabled=True
        )
        == 0.8
    )
    assert (
        ball_priority_clearance_m(agent_id="red.defender", priority_agent_id=None, enabled=True)
        == 0.8
    )
    assert (
        ball_priority_clearance_m(
            agent_id="red.defender", priority_agent_id="red.defender", enabled=False
        )
        == 0.8
    )


@pytest.mark.parametrize(
    "agent,priority,enabled", [("", None, True), ("red", "", True), ("red", None, 1)]
)
def test_malformed_priority_is_rejected(agent, priority, enabled):
    with pytest.raises(ValueError):
        ball_priority_clearance_m(agent_id=agent, priority_agent_id=priority, enabled=enabled)


def test_priority_changes_projection_but_retains_neighbor_constraint():
    nominal = np.array([0.7, 0.0])
    neighbor = np.array([[0.7, 0.0]])
    ordinary = propose_clearance_velocity(nominal, neighbor, maximum_speed_mps=0.7, clearance_m=0.8)
    priority = propose_clearance_velocity(nominal, neighbor, maximum_speed_mps=0.7, clearance_m=0.6)
    assert ordinary.velocity_mps[0] < 0
    assert 0 < priority.velocity_mps[0] < nominal[0]
    assert priority.feasible and priority.constrained


def test_opt_in_requires_clearance_and_preserves_disabled_config_identity():
    world = replace(
        default_continuous_match_config(), bilateral_goals=True, stop_on_ball_exit=False
    )
    assert (
        world.config_hash
        == "sha256:0e1b88af2d9a1763da3abaea2640140d379e4a5ee5799d107e7e5c4f07901733"
    )
    with pytest.raises(ValueError):
        replace(world, ball_priority_clearance=True)
    enabled = replace(world, all_role_clearance=True, ball_priority_clearance=True)
    assert enabled.config_hash != replace(world, all_role_clearance=True).config_hash
