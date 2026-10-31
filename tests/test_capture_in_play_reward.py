import numpy as np
import pytest
from test_s215_near_ball_residual import samples

from rosclaw_soccer.growth.near_ball_residual import NearBallResidualPolicy
from rosclaw_soccer.growth.role_self_model import TacticalIntent
from rosclaw_soccer.training.in_play_rewards import in_play_rewards
from rosclaw_soccer.training.near_ball_residual_ppo import physical_rewards, update_private_actors


def fixture():
    ids = tuple(f"{team}.{i}" for team in ("blue", "red") for i in range(4))
    parent = NearBallResidualPolicy.initialize(ids, "sha256:" + "a" * 64)
    trace = samples(parent)
    n = len(trace["time"])
    trace["ball_pose"][:, 2:4] = [0.11, 1.0]
    trace["pitch_boundary_geometry"] = np.tile([-1.5, 7.5, 3.0, 0.11, 3.0, 2.0], (n, 1))
    trace["ball_control_entry_velocity"] = np.zeros((n, 6))
    trace["ball_nonfoot_contact_force_n"] = np.zeros(n)
    trace["robot_robot_contact_count"] = np.zeros(n)
    trace["possession_agent_code"] = np.zeros(n, dtype=int)
    for agent in ids:
        key = agent.replace(".", "_")
        trace[key + "_intent_code"] = np.full(n, list(TacticalIntent).index(TacticalIntent.RECEIVE))
        trace[key + "_joint_safety_margin_rad"] = np.full((n, 29), 0.1)
        trace[key + "_pelvis_pose"][:, 2:] = [0.75, 1, 0, 0, 0]
    return ids, trace


@pytest.mark.parametrize("terminal", ["in_play", "out", "goal"])
def test_composition_preserves_capture_and_applies_match_terminal(terminal):
    ids, trace = fixture()
    trace["ball_contact_agent_code"][0] = 5
    trace["ball_contact_force_n"][0] = 2
    if terminal == "out":
        trace["ball_pose"][-1, 1] = 3.2
    elif terminal == "goal":
        trace["ball_pose"][-1, 0] = 7.7
    before = {k: v.copy() for k, v in trace.items()}
    base = physical_rewards(trace, ids, reward_shaping="team_capture_dissipation_v1")
    expected = in_play_rewards(base, trace, ids)
    actual = physical_rewards(trace, ids, reward_shaping="team_capture_dissipation_in_play_v1")
    np.testing.assert_array_equal(actual, expected)
    for key in trace:
        np.testing.assert_array_equal(trace[key], before[key])


def test_rejects_post_boundary_observations_even_if_ball_reenters():
    ids, trace = fixture()
    trace["ball_pose"][-2, 1] = 3.2
    with pytest.raises(ValueError, match="terminate at the first boundary"):
        physical_rewards(trace, ids, reward_shaping="team_capture_dissipation_in_play_v1")


def test_missing_boundary_evidence_fails_closed():
    ids, trace = fixture()
    del trace["pitch_boundary_geometry"]
    with pytest.raises(ValueError, match="geometry"):
        physical_rewards(trace, ids, reward_shaping="team_capture_dissipation_in_play_v1")


def test_new_objective_drives_on_policy_private_actor_update():
    pytest.importorskip("torch")
    ids, trace = fixture()
    parent = NearBallResidualPolicy.initialize(ids, "sha256:" + "a" * 64)
    trace["ball_pose"][-1, 1] = 3.2
    child, rows = update_private_actors(
        parent, [trace], reward_shaping="team_capture_dissipation_in_play_v1"
    )
    assert child.parent_hash == parent.policy_hash
    assert rows[0]["updated"]
    assert not any(row["updated"] for row in rows[1:])
