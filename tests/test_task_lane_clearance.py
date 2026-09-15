import numpy as np
import pytest

from rosclaw_soccer.world.player_clearance import propose_clearance_velocity
from rosclaw_soccer.world.task_lane_clearance import (
    stance_clearance_halfplane,
    stance_reservation_allowed,
)


@pytest.mark.parametrize("intent", ["support", "cover", "run_in_behind"])
def test_soft_reservation_yields_to_receive_protection(intent):
    assert stance_reservation_allowed(intent=intent, post_receive_hold=False)
    assert not stance_reservation_allowed(intent=intent, post_receive_hold=True)


@pytest.mark.parametrize(
    "intent", ["receive", "recover", "save", "pass", "shoot", "hold", "unknown"]
)
def test_soft_reservation_never_claims_an_active_or_unknown_skill(intent):
    assert not stance_reservation_allowed(intent=intent, post_receive_hold=False)


def test_reservation_moves_teammate_outward_without_removing_neighbor_guards():
    player = np.array([4.07, 0.23])
    stance = np.array([3.71, 0.126])
    plane = stance_clearance_halfplane(player, stance, np.array([1.0, 0]), clearance_m=0.9)
    neighbors = np.array([[5.03, -0.12], [3.51, -0.44]]) - player
    result = propose_clearance_velocity(
        np.zeros(2), neighbors, maximum_speed_mps=0.7, additional_halfplanes=plane
    )
    assert result.feasible
    v = np.array(result.velocity_mps)
    assert (plane[:, :2] @ v >= plane[:, 2] - 1e-9).all()
    for neighbor in neighbors:
        d = np.linalg.norm(neighbor)
        assert (-neighbor / d) @ v >= 0.6 * (0.8 - d) - 1e-9


def test_half_turn_symmetry_and_no_change_outside_reserved_stance():
    args = (np.array([0.2, 0.1]), np.zeros(2), np.array([1.0, 0]))
    red = stance_clearance_halfplane(*args, clearance_m=0.9)
    blue = stance_clearance_halfplane(*(-v for v in args), clearance_m=0.9)
    np.testing.assert_array_equal(red[:, :2], -blue[:, :2])
    np.testing.assert_array_equal(red[:, 2], blue[:, 2])
    assert stance_clearance_halfplane(np.array([2.0, 0]), args[1], args[2], clearance_m=0.9) is None


def test_exact_overlap_uses_declared_direction_and_invalid_inputs_fail_closed():
    plane = stance_clearance_halfplane(
        np.zeros(2), np.zeros(2), np.array([0.0, -1]), clearance_m=0.9
    )
    np.testing.assert_array_equal(plane, [[0.0, -1.0, 0.4]])
    for radius in (True, 0, 2, float("nan")):
        with pytest.raises(ValueError, match="task-space"):
            stance_clearance_halfplane(np.zeros(2), np.zeros(2), np.ones(2), clearance_m=radius)


def test_task_approach_config_requires_simultaneous_clearance_and_preserves_old_identity():
    from rosclaw_soccer.skills.team.independent_team_world import IndependentTeamWorldConfig

    old = IndependentTeamWorldConfig()
    new = IndependentTeamWorldConfig(
        prospective_strike_approach=True,
        all_role_clearance=True,
        teammate_approach_clearance_m=0.9,
    )
    assert old.config_hash != new.config_hash
    for kwargs in (
        {"teammate_approach_clearance_m": 0.9},
        {"prospective_strike_approach": True, "teammate_approach_clearance_m": 0.9},
        {"prospective_strike_approach": 1},
        {
            "prospective_strike_approach": True,
            "all_role_clearance": True,
            "teammate_approach_clearance_m": float("nan"),
        },
    ):
        with pytest.raises(ValueError, match="task approach"):
            IndependentTeamWorldConfig(**kwargs)
