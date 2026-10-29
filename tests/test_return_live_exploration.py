from dataclasses import replace

import pytest

from rosclaw_soccer.growth.near_ball_residual import NearBallResidualPolicy
from rosclaw_soccer.skills.team.independent_team_world import (
    _returned_live_exploration_state,
    _returned_live_policy,
)


def test_exploration_starts_only_at_reentry_and_stops_at_exit():
    state = False
    observed = []
    for code in (0, 1, 0, 2, 0, 3, 0, 1, 0, 2, 3):
        state = _returned_live_exploration_state(state, code)
        observed.append(state)
    assert observed == [False, False, False, False, False, True, True, False, False, False, True]


@pytest.mark.parametrize("state,event", [(1, 0), (False, True), (False, 4), (False, -1)])
def test_rejects_invalid_return_lifecycle_inputs(state, event):
    with pytest.raises(ValueError):
        _returned_live_exploration_state(state, event)


def test_policy_switch_requires_exact_lineage_and_return_event():
    parent = NearBallResidualPolicy.initialize(
        tuple(f"a{i}" for i in range(8)), "sha256:" + "a" * 64
    )
    candidate = replace(parent, generation=1, parent_hash=parent.policy_hash)
    assert _returned_live_policy(parent, candidate, False) is parent
    assert _returned_live_policy(parent, candidate, True) is candidate
    with pytest.raises(ValueError):
        _returned_live_policy(parent, replace(candidate, parent_hash="sha256:" + "b" * 64), True)
    with pytest.raises(ValueError):
        _returned_live_policy(parent, candidate, 1)
