"""Targeted checks for pass-target-aware phase stance/orientation injection.

The default (goal-referenced) path must stay bit-identical to the frozen S208
behavior; only an explicit PASS decision target may redirect the phase metrics.
"""

import math
from types import SimpleNamespace

import numpy as np
import pytest

from rosclaw_soccer.growth.independent_agent_cell import (
    AgentCellDecision,
    SoccerSkill,
    TacticalIntent,
)
from rosclaw_soccer.growth.strike_phase_controller import StrikePhaseConfig
from rosclaw_soccer.skills.team.independent_team_world import (
    _controller_strike_stance_metrics,
    _phase_task_target,
    _strike_tracking_yaw_error,
)
from rosclaw_soccer.world.field import G1TrainingGoalSpec

_HASH = "sha256:" + "0" * 64


def _decision(intent: TacticalIntent, target: tuple[float, float, float]) -> AgentCellDecision:
    return AgentCellDecision(
        agent_id="red.finisher",
        intent=intent,
        skill=SoccerSkill.LEAD_PASS if intent is TacticalIntent.PASS else SoccerSkill.FINISHING,
        target_position_m=target,
        target_agent_id="red.playmaker" if intent is TacticalIntent.PASS else None,
        confidence=0.5,
        observation_hash=_HASH,
        policy_artifact_hash=_HASH,
    )


def _fake_world():
    qpos = np.zeros(43)
    qpos[0:3] = (2.0, 1.0, 0.9)  # pelvis
    qpos[3] = 1.0  # unit quaternion (w,x,y,z) -> yaw 0
    qpos[36:39] = (3.0, 1.0, 0.115)  # ball
    qvel = np.zeros(41)
    data = SimpleNamespace(qpos=qpos, qvel=qvel)
    controller = SimpleNamespace(qpos_base=0)
    return controller, data


def test_phase_task_target_only_for_pass():
    controller = SimpleNamespace(decision=None)
    assert _phase_task_target(controller) is None
    controller = SimpleNamespace(decision=_decision(TacticalIntent.SHOOT, (5.0, 1.0, 0.2)))
    assert _phase_task_target(controller) is None
    controller = SimpleNamespace(decision=_decision(TacticalIntent.PASS, (3.0, 3.0, 0.2)))
    assert _phase_task_target(controller) == (3.0, 3.0)


def test_stance_metrics_default_goal_path_unchanged():
    controller, data = _fake_world()
    goal = G1TrainingGoalSpec()
    depth, lateral, yaw = _controller_strike_stance_metrics(
        controller=controller, data=data, ball_qpos=36, goal=goal
    )
    assert depth == pytest.approx(1.0)
    assert lateral == pytest.approx(0.0)
    assert yaw == pytest.approx(0.0)


def test_stance_metrics_injected_pass_target():
    controller, data = _fake_world()
    goal = G1TrainingGoalSpec()
    depth, lateral, yaw = _controller_strike_stance_metrics(
        controller=controller, data=data, ball_qpos=36, goal=goal, target_xy=(3.0, 3.0)
    )
    assert depth == pytest.approx(0.0)
    assert lateral == pytest.approx(1.0)
    assert yaw == pytest.approx(math.pi / 2)


def test_tracking_yaw_error_target_changes_value_without_mutation():
    controller, data = _fake_world()
    before_pos = data.qpos.copy()
    before_vel = data.qvel.copy()
    goal = G1TrainingGoalSpec()
    config = StrikePhaseConfig()
    goal_path = _strike_tracking_yaw_error(
        controller=controller, data=data, ball_qpos=36, ball_qvel=35, goal=goal, config=config
    )
    task_path = _strike_tracking_yaw_error(
        controller=controller,
        data=data,
        ball_qpos=36,
        ball_qvel=35,
        goal=goal,
        config=config,
        target_xy=(3.0, 3.0),
    )
    assert goal_path != pytest.approx(task_path)
    np.testing.assert_array_equal(data.qpos, before_pos)
    np.testing.assert_array_equal(data.qvel, before_vel)
