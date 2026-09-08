from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from rosclaw_soccer.growth.owned_ball_contact import OwnedBallContactPolicy
from rosclaw_soccer.growth.role_self_model import MatchRole, TacticalIntent
from rosclaw_soccer.skills.team.independent_team_world import (
    IndependentTeamWorldConfig,
    _movement_command,
)
from rosclaw_soccer.training.active_team_probe import run_probe
from rosclaw_soccer.training.near_ball_residual_ppo import physical_rewards


def test_anticipatory_planning_does_not_require_neural_kick_option(tmp_path: Path) -> None:
    # The already-existing output is rejected before loading any external assets.
    # Previously this valid composition incorrectly failed its option dependency.
    with pytest.raises(FileExistsError):
        run_probe(
            asset_root=tmp_path,
            output=tmp_path,
            active=True,
            duration=5,
            four_vs_four=True,
            anticipatory_contact=True,
            bilateral_kick_options=False,
        )


def test_prospective_near_ball_stance_advances_without_claiming_possession() -> None:
    controller = SimpleNamespace(
        cell=SimpleNamespace(
            agent_id="red.playmaker",
            self_model=SimpleNamespace(
                team_id="red", opponent_ids=(), teammate_ids=(), primary_role=MatchRole.PLAYMAKER
            ),
        ),
        qpos_base=0,
        left_ankle_body=0,
        right_ankle_body=1,
        last_world_command=np.zeros(3),
    )
    data = SimpleNamespace(
        qpos=np.array([0, 0, 0.75, 1, 0, 0, 0, 0.5, 0, 0.115]),
        qvel=np.zeros(12),
        xpos=np.array([[0, 0.1, 0.03], [0, -0.1, 0.03]]),
    )
    kwargs = dict(
        controller=controller,
        decision=SimpleNamespace(intent=TacticalIntent.PASS, target_position_m=(2.0, 0.0, 0.0)),
        positions={"red.playmaker": np.zeros(2)},
        data=data,
        ball_qpos=7,
        ball_qvel=6,
        committed_receiver=False,
        active_receiver=False,
        post_receive_hold=False,
        receive_foot_lateral_offset_m=0.18,
        strike_target_position_m=None,
        config=IndependentTeamWorldConfig(owned_contact_policy=OwnedBallContactPolicy()),
    )
    old = _movement_command(**kwargs, possession_agent_id=None)
    planned = _movement_command(**kwargs, possession_agent_id=None, prospective_contact=True)
    opponent = _movement_command(
        **kwargs, possession_agent_id="blue.finisher", prospective_contact=True
    )
    assert old[0] < 0 < planned[0]
    assert opponent[0] < 0


def test_team_reward_requires_completed_physical_pass_and_is_delayed() -> None:
    ids = tuple(f"agent.{i}" for i in range(8))
    trace = {
        "time": np.array([0.02, 0.04, 0.06]),
        "residual_observations": np.zeros((3, 8, 56)),
        "residual_applied": np.zeros((3, 8, 12)),
        "ball_pose": np.array([[0.0, 0.0, 0.1], [0.4, 0.0, 0.1], [0.8, 0.0, 0.1]]),
        "ball_velocity": np.zeros((3, 6)),
        "ball_contact_agent_code": np.array([1, 0, 2]),
        "ball_contact_effector_code": np.array([1, 0, 1]),
        "ball_contact_force_n": np.array([10.0, 0.0, 10.0]),
    }
    for key in (
        "pass_source_agent_code",
        "pass_target_agent_code",
        "ball_nonfoot_contact_agent_code",
        "robot_robot_contact_first_code",
        "robot_robot_contact_second_code",
    ):
        trace[key] = np.zeros(3)
    for agent in ids:
        key = agent.replace(".", "_")
        trace[key + "_pelvis_pose"] = np.ones((3, 7))
        for suffix in ("_left_foot_position", "_right_foot_position", "_target_position"):
            trace[key + suffix] = np.zeros((3, 3))
    unplanned = physical_rewards(trace, ids)
    trace["pass_source_agent_code"][:] = 1
    trace["pass_target_agent_code"][:] = 2
    planned = physical_rewards(trace, ids)
    delta = planned - unplanned
    np.testing.assert_array_equal(delta[:2], np.zeros((2, 8)))
    np.testing.assert_allclose(delta[2], [1, 1, 0, 0, 0, 0, 0, 0])
    trace["ball_contact_force_n"][2] = 0
    no_receive = physical_rewards(trace, ids)
    np.testing.assert_allclose(no_receive, unplanned)
