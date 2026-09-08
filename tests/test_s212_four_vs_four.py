from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from rosclaw_soccer.growth.independent_agent_cell import (
    AgentCellObservation,
    AgentPhysicalState,
)
from rosclaw_soccer.growth.role_self_model import MatchRole, TacticalIntent
from rosclaw_soccer.training.four_vs_four_match import build_four_vs_four_fixture
from rosclaw_soccer.world.bilateral_net import apply_opposite_goal_net_force
from rosclaw_soccer.world.field import (
    G1CompliantGoalNetState,
    G1TrainingGoalSpec,
    apply_g1_compliant_goal_net_force,
)


@pytest.fixture
def fixture(monkeypatch):
    import rosclaw_soccer.training.four_vs_four_match as module
    from rosclaw_soccer.growth.independent_agent_cell import build_independent_agent_cell
    from rosclaw_soccer.sim.contracts import hash_json

    digest = hash_json({"fixture": "s212"})
    cell = build_independent_agent_cell(
        agent_id="red.goalkeeper",
        team_id="red",
        primary_role=MatchRole.GOALKEEPER,
        teammate_ids=("red.playmaker",),
        opponent_ids=("blue.playmaker",),
        body_hash=digest,
        foundation_policy_hash=digest,
        home_position_m=(0.0, 0.0, 0.0),
    )
    monkeypatch.setattr(
        module,
        "build_independent_three_vs_three_fixture",
        lambda _: SimpleNamespace(
            cells=(cell,),
            foundation_policy_hash=digest,
            goal=G1TrainingGoalSpec(plane_x_m=7.5, width_m=3.0, height_m=2.0),
        ),
    )
    return build_four_vs_four_fixture(Path("unused"))


def test_role_complete_mirrored_roster_and_private_memories(fixture):
    assert len(fixture.cells) == 8
    assert len({c.growth_scope.personal_memory_namespace for c in fixture.cells}) == 8
    for role in MatchRole:
        red = next(p for p in fixture.players if p.agent_id == "red." + role.value)
        blue = next(p for p in fixture.players if p.agent_id == "blue." + role.value)
        assert blue.origin_m == (6.0 - red.origin_m[0], -red.origin_m[1], 0.0)
    with pytest.raises(ValueError):
        replace(fixture, players=(*fixture.players, fixture.players[0]))


def test_forward_lane_curriculum_is_mirrored_and_distinct_from_baseline(fixture):
    curriculum = build_four_vs_four_fixture(Path("unused"), forward_receiver_lane=True)
    assert curriculum.fixture_hash != fixture.fixture_hash
    red = next(p for p in curriculum.players if p.agent_id == "red.finisher")
    blue = next(p for p in curriculum.players if p.agent_id == "blue.finisher")
    assert red.origin_m == (3.5, -1.1, 0.0)
    assert blue.origin_m == (2.5, 1.1, 0.0)


def _observation(fixture, agent_id, *, blue=False):
    def state(p):
        return AgentPhysicalState(
            p.agent_id, (*p.origin_m[:2], 0.78), (0.0, 0.0, 0.0), 0.78, 0.02, True
        )

    states = {p.agent_id: state(p) for p in fixture.players}
    cell = next(c for c in fixture.cells if c.agent_id == agent_id)
    return AgentCellObservation(
        observer_agent_id=agent_id,
        time_sec=0.0,
        ball_position_m=(4.0, 1.2, 0.115) if blue else (2.0, -1.2, 0.115),
        ball_velocity_mps=(0.0, 0.0, 0.0),
        own_goal_m=(7.5, 0.0, 0.0) if agent_id.startswith("blue") else (-1.5, 0.0, 0.0),
        opponent_goal_m=(-1.5, 0.0, 0.0) if agent_id.startswith("blue") else (7.5, 0.0, 0.0),
        possession_agent_id=None,
        self_state=states[agent_id],
        teammate_states=tuple(states[i] for i in cell.self_model.teammate_ids),
        opponent_states=tuple(states[i] for i in cell.self_model.opponent_ids),
    )


def test_mirrored_roles_make_mirrored_decisions(fixture):
    for role in MatchRole:
        red = next(c for c in fixture.cells if c.agent_id == "red." + role.value)
        blue = next(c for c in fixture.cells if c.agent_id == "blue." + role.value)
        r = red.decide(_observation(fixture, red.agent_id))
        b = blue.decide(_observation(fixture, blue.agent_id, blue=True))
        assert r.intent == b.intent
        assert b.target_position_m == pytest.approx(
            (6.0 - r.target_position_m[0], -r.target_position_m[1], r.target_position_m[2])
        )


def test_defender_protects_and_each_team_has_a_challenger(fixture):
    decisions = [c.decide(_observation(fixture, c.agent_id)) for c in fixture.cells]
    for team in ("red", "blue"):
        own = [d for d in decisions if d.agent_id.startswith(team + ".")]
        assert sum(d.intent == TacticalIntent.RECEIVE for d in own) == 1
        assert (
            next(d for d in own if d.agent_id.endswith(".defender")).intent == TacticalIntent.COVER
        )


def test_opposite_net_is_identical_under_rotation_without_pose_writes():
    spec = G1TrainingGoalSpec(plane_x_m=7.5, width_m=3.0, height_m=2.0)
    right = SimpleNamespace(
        qpos=np.array([8.8, 0.2, 0.5]),
        qvel=np.array([3.0, 0.1, -0.2]),
        xfrc_applied=np.zeros((1, 6)),
    )
    left = SimpleNamespace(
        qpos=np.array([-2.8, -0.2, 0.5]),
        qvel=np.array([-3.0, -0.1, -0.2]),
        xfrc_applied=np.zeros((1, 6)),
    )
    before = left.qpos.copy(), left.qvel.copy()
    rstate, lstate = G1CompliantGoalNetState(), G1CompliantGoalNetState()
    for _ in range(2):
        left.xfrc_applied[:] = 0.0
        apply_g1_compliant_goal_net_force(
            right,
            ball_body_id=0,
            ball_qpos=0,
            ball_qvel=0,
            spec=spec,
            capture_depth_m=0.8 * spec.depth_m,
            stiffness_n_m=180.0,
            damping_n_s_m=10.0,
            state=rstate,
        )
        apply_opposite_goal_net_force(
            left,
            ball_body_id=0,
            ball_qpos=0,
            ball_qvel=0,
            spec=spec,
            left_goal_plane_x_m=-1.5,
            state=lstate,
        )
        assert left.xfrc_applied[0, :3] == pytest.approx(
            right.xfrc_applied[0, :3] * [-1.0, -1.0, 1.0]
        )
    np.testing.assert_array_equal(left.qpos, before[0])
    np.testing.assert_array_equal(left.qvel, before[1])
    assert rstate.engaged and lstate.engaged


def test_both_goal_frames_have_mirrored_physical_geometry():
    import mujoco

    from rosclaw_soccer.world.field import _add_goal

    parent = mujoco.MjSpec()
    spec = G1TrainingGoalSpec(plane_x_m=7.5, width_m=3.0, height_m=2.0)
    _add_goal(parent, spec)
    _add_goal(parent, spec, mirror_sum_x=6.0)
    model = parent.compile()
    for index in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, index)
        if name.startswith("opposite_"):
            continue
        opposite = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "opposite_" + name)
        assert opposite >= 0
        x, y, z = model.geom_pos[index]
        assert model.geom_pos[opposite] == pytest.approx((6.0 - x, -y, z))
        np.testing.assert_array_equal(model.geom_size[index], model.geom_size[opposite])
        assert model.geom_contype[index] == model.geom_contype[opposite]


def test_blue_strike_stance_uses_blue_attacking_direction():
    from rosclaw_soccer.skills.team.independent_team_world import _strike_teacher_stance_ready

    data = SimpleNamespace(qpos=np.array([4.4, 0.0, 0.78, 4.0, 0.0, 0.115]))
    controller = SimpleNamespace(qpos_base=0)
    goal = G1TrainingGoalSpec(plane_x_m=7.5)
    assert _strike_teacher_stance_ready(
        controller=controller,
        data=data,
        ball_qpos=3,
        goal=goal,
        minimum_depth_m=0.3,
        target_xy=(-1.5, 0.0),
    )
    assert not _strike_teacher_stance_ready(
        controller=controller,
        data=data,
        ball_qpos=3,
        goal=goal,
        minimum_depth_m=0.3,
        target_xy=(7.5, 0.0),
    )


def test_nearest_contact_lease_is_not_monopolized_by_distant_chaser():
    from rosclaw_soccer.growth.locomotion_contact_teacher import G1LocomotionContactTeacherConfig
    from rosclaw_soccer.skills.team.independent_team_world import _select_contact_teacher_controller

    controllers = tuple(
        SimpleNamespace(
            cell=SimpleNamespace(agent_id=agent),
            decision=SimpleNamespace(intent=TacticalIntent.RECEIVE),
            left_ankle_body=2 * i,
            right_ankle_body=2 * i + 1,
        )
        for i, agent in enumerate(("red.playmaker", "blue.playmaker"))
    )
    data = SimpleNamespace(
        xpos=np.array([[0.5, 0.0, 0.0], [0.5, 0.1, 0.0], [0.1, 0.0, 0.0], [0.1, 0.1, 0.0]])
    )
    selected = _select_contact_teacher_controller(
        controllers=controllers,
        data=data,
        ball_position=np.zeros(3),
        current_possession_agent_id=None,
        preferred_agent_id="red.playmaker",
        config=G1LocomotionContactTeacherConfig(),
        nearest_contact_first=True,
    )
    assert selected.cell.agent_id == "blue.playmaker"


@pytest.mark.integration
def test_physical_eight_g1_model_has_two_goals_and_232_actuators():
    import os

    import mujoco

    from rosclaw_soccer.world.multi_player import build_g1_multi_player_stadium_model

    root = os.environ.get("ROSCLAW_G1_ASSET_ROOT")
    if root is None:
        pytest.skip("ROSCLAW_G1_ASSET_ROOT is not configured")
    fixture = build_four_vs_four_fixture(Path(root))
    model = build_g1_multi_player_stadium_model(
        Path(root),
        players=fixture.players,
        spec=fixture.goal,
        left_goal_plane_x_m=-1.5,
    )
    assert model.nu == 232
    for name in ("goal_left_post", "opposite_goal_left_post"):
        geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        assert geom >= 0 and model.geom_contype[geom] != 0


@pytest.mark.parametrize(
    "position,reason",
    [
        ((3.0, 3.10, 0.115), None),
        ((3.0, 3.12, 0.115), "TOUCHLINE_OUT"),
        ((7.62, 0.0, 0.5), "RED_GOAL_CROSSING"),
        ((-1.62, 0.0, 0.5), "BLUE_GOAL_CROSSING"),
        ((7.62, 2.0, 0.5), "GOAL_LINE_OUT"),
        ((7.62, 0.0, 2.1), "GOAL_LINE_OUT"),
        ((float("nan"), 0.0, 0.5), "NONFINITE_STATE"),
    ],
)
def test_whole_ball_exit_reason(position, reason):
    from rosclaw_soccer.world.match_boundary import ball_exit_reason

    assert (
        ball_exit_reason(
            position, left_x=-1.5, right_x=7.5, radius=0.115, goal_width=3.0, goal_height=2.0
        )
        == reason
    )
