from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from rosclaw_soccer.growth.locomotion_contact_teacher import G1RollingOptionBridgeConfig
from rosclaw_soccer.growth.role_self_model import TacticalIntent
from rosclaw_soccer.skills.team.independent_team_world import (
    _activate_rolling_option,
    _near_ball_observation,
)
from rosclaw_soccer.world.field import G1TrainingGoalSpec


def controller_fixture(*, blue=False, y=0.45, intent=TacticalIntent.SHOOT):
    agent = "blue.finisher" if blue else "red.finisher"
    controller = SimpleNamespace(
        spec=SimpleNamespace(yaw_rad=np.pi if blue else 0.0),
        cell=SimpleNamespace(
            agent_id=agent, self_model=SimpleNamespace(team_id="blue" if blue else "red")
        ),
        decision=SimpleNamespace(
            intent=intent, target_position_m=(-1.5, 0.0, 1.0) if blue else (7.5, 0.0, 1.0)
        ),
        option_active=False,
        option_completed=False,
        qpos_base=0,
        kick_policy=SimpleNamespace(enter=lambda: None, WARMUP_STEPS=50),
        kick_output=SimpleNamespace(),
    )
    qpos = np.array(
        [3.5, -y, 0.75, 0, 0, 0, 1, 3, 0, 0.115]
        if blue
        else [2.5, y, 0.75, 1, 0, 0, 0, 3, 0, 0.115]
    )
    return controller, SimpleNamespace(qpos=qpos)


def activate(controller, data, config):
    _activate_rolling_option(
        controllers=(controller,),
        current_possession_agent_id=controller.cell.agent_id,
        last_ball_contact_agent_id=controller.cell.agent_id,
        strike_lease_agent_id=None,
        frame=1,
        data=data,
        ball_qpos=7,
        goal=replace(
            G1TrainingGoalSpec(), plane_x_m=7.5, height_m=2.0, target_y_m=0.8, target_z_m=1.5
        ),
        config=config,
        left_goal_plane_x_m=-1.5,
    )


@pytest.mark.parametrize("blue", [False, True])
@pytest.mark.parametrize("bound", [False, True])
def test_admission_checks_the_actual_motor_target_not_an_easier_planner_line(blue, bound):
    controller, data = controller_fixture(blue=blue)
    activate(
        controller,
        data,
        G1RollingOptionBridgeConfig(
            bilateral_enabled=True,
            minimum_strike_stance_depth_m=0.3,
            maximum_strike_lateral_error_m=0.5,
            task_context_bound=bound,
        ),
    )
    assert controller.option_active is (not bound)


def test_reference_conditioning_does_not_replace_the_pass_task_target():
    controller, data = controller_fixture(y=0, intent=TacticalIntent.PASS)
    controller.decision.target_position_m = (5.0, 0.0, 0.115)
    activate(
        controller,
        data,
        G1RollingOptionBridgeConfig(
            pass_enabled=True,
            pass_reference_distance_m=4.0,
            task_context_bound=True,
        ),
    )
    assert controller.option_active
    assert controller.option_task_target_m == (5.0, 0.0, 0.115)
    assert controller.kick_policy.target_pos_w[0] == 6.5
    controller.decision.target_position_m = (0.0, 0.0, 0.0)
    assert controller.option_task_target_m == (5.0, 0.0, 0.115)


def test_proprioception_task_direction_matches_the_admitted_option():
    controller = SimpleNamespace(
        state=SimpleNamespace(
            pelvis_quat_w=np.array([1.0, 0, 0, 0]),
            pelvis_pos_w=np.array([2.5, 0.0, 0.75]),
            gravity_ori=np.array([0.0, 0.0, -1.0]),
            ang_vel=np.zeros(3),
            q=np.zeros(29),
            dq=np.zeros(29),
        ),
        decision=SimpleNamespace(target_position_m=(2.9, 0.0, 0.0)),
        policy=SimpleNamespace(default_angles_reorder=np.zeros(29)),
        left_ankle_body=0,
        right_ankle_body=1,
    )
    kwargs = dict(
        data=SimpleNamespace(
            qpos=np.array([3.0, 0.0, 0.115]), qvel=np.zeros(3), xpos=np.zeros((2, 3))
        ),
        ball_qpos=0,
        ball_qvel=0,
        previous=np.zeros(12),
    )
    old = _near_ball_observation(controller, **kwargs)
    bound = _near_ball_observation(controller, **kwargs, task_target_position_m=(7.5, 0.8, 1.5))
    assert old[36] == -1
    assert bound[36] > 0.9
    assert bound[37] > 0
    np.testing.assert_array_equal(old[:36], bound[:36])
    np.testing.assert_array_equal(old[38:], bound[38:])


def test_bound_context_is_explicit_hash_bound_and_keeps_legacy_hash():
    old = G1RollingOptionBridgeConfig()
    assert (
        old.config_hash == "sha256:f455e7d360bd866fba5d2f8b061b68440b97408aef2556fbdcae99f9c3fa43de"
    )
    assert replace(old, task_context_bound=True).config_hash != old.config_hash
    with pytest.raises(ValueError, match="SIM-only"):
        G1RollingOptionBridgeConfig(task_context_bound=1)
