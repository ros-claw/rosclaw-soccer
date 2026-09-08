from dataclasses import replace

import numpy as np
import pytest

from rosclaw_soccer.growth.contact_stroke import ContactStroke, stroke_blend
from rosclaw_soccer.growth.locomotion_contact_teacher import G1LocomotionContactTeacherConfig


def test_stroke_smooth_monotone_and_zero_endpoint_velocity() -> None:
    assert stroke_blend(0) == (0, 0)
    assert stroke_blend(1) == (1, 0)
    samples = [stroke_blend(float(x)) for x in np.linspace(0, 1, 101)]
    assert np.all(np.diff([p for p, _ in samples]) >= 0)
    assert all(v >= 0 for _, v in samples)


def test_stroke_locks_foot_expires_without_rearming() -> None:
    stroke = ContactStroke()
    assert stroke.step(time_sec=2, duration_sec=0.3, use_left=True) == 0
    assert stroke.step(time_sec=2.15, duration_sec=0.3, use_left=False) == pytest.approx(0.5)
    assert stroke.use_left
    assert stroke.step(time_sec=3, duration_sec=0.3, use_left=False) == 1
    assert stroke.step(time_sec=4, duration_sec=0.3, use_left=False) == 1
    stroke.reset()
    assert stroke.step(time_sec=4, duration_sec=0.3, use_left=False) == 0
    assert not stroke.use_left


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -0.1, 1.1])
def test_invalid_progress(value: float) -> None:
    with pytest.raises(ValueError):
        stroke_blend(value)


def test_default_disabled_and_duration_bounded() -> None:
    config = G1LocomotionContactTeacherConfig()
    assert config.pass_stroke_duration_sec == 0
    for value in (-1, 0.1, 0.61, float("nan")):
        with pytest.raises(ValueError):
            replace(config, pass_stroke_duration_sec=value)
    for value in (0.12, 0.3, 0.6):
        assert replace(config, pass_stroke_duration_sec=value).training_only


def test_direction_and_lateral_sign_do_not_jump_during_stroke() -> None:
    stroke = ContactStroke()
    stroke.step(
        time_sec=1, duration_sec=0.3, use_left=True, direction_xy=(0.0, 2.0), lateral_sign=-1.0
    )
    stroke.step(
        time_sec=1.1, duration_sec=0.3, use_left=False, direction_xy=(2.0, 0.0), lateral_sign=1.0
    )
    assert stroke.direction_xy == (0.0, 1.0)
    assert stroke.lateral_sign == -1.0
    assert stroke.use_left
    with pytest.raises(ValueError, match="backwards"):
        stroke.step(time_sec=1.05, duration_sec=0.3, use_left=True)


def test_warmstart_changes_reference_not_measured_state(monkeypatch: pytest.MonkeyPatch) -> None:
    from types import SimpleNamespace

    import rosclaw_soccer.providers.g1.kick_warmstart as module

    monkeypatch.setattr(
        module.importlib,
        "import_module",
        lambda _: SimpleNamespace(NPZ_ANCHOR_IDX=0, ISAAC_TO_MUJOCO=np.arange(29)),
    )
    q = np.arange(29, dtype=float) / 100
    state = SimpleNamespace(q=q.copy())
    policy = SimpleNamespace(
        use_body_frame_ball=False,
        runtime_mode="sim",
        motion_body_pos=np.asarray([[[0.0, 0.0, 1.0]], [[0.5, -0.2, 1.0]]]),
        action_scale_mj=np.ones(29),
        default_q_mj=np.zeros(29),
        state_cmd=state,
        _init_to_world=np.eye(3),
        action_clip_lo_il=-np.ones(29),
        action_clip_hi_il=np.ones(29),
        _build_obs=lambda: np.zeros(547),
    )
    module.prepare_kick_handoff(policy, entry_frame=1)
    np.testing.assert_array_equal(state.q, q)
    np.testing.assert_array_equal(policy._ref_anchor_world_origin, [0.5, -0.2, 1.0])
    np.testing.assert_allclose(policy.last_action_il, q)
    policy.runtime_mode = "real"
    with pytest.raises(ValueError, match="simulation-only"):
        module.prepare_kick_handoff(policy, entry_frame=1)


def test_completion_has_no_residual_and_old_touch_does_not_skip_windup() -> None:
    from types import SimpleNamespace

    from rosclaw_soccer.growth.locomotion_contact_teacher import locomotion_contact_teacher_effect

    config = G1LocomotionContactTeacherConfig(pass_stroke_duration_sec=0.3)
    data = SimpleNamespace(xpos=np.asarray([[0.0, 0.0, 0.1]]), qvel=np.zeros(32))

    def effect(ball_x: float, progress: float):
        return locomotion_contact_teacher_effect(
            model=SimpleNamespace(nv=32),
            data=data,
            ankle_body_id=0,
            actuated_dof_indices=np.arange(3, 32),
            ball_position_m=np.asarray([ball_x, 0.0, 0.115]),
            ball_velocity_mps=np.zeros(3),
            desired_ball_direction_xy=np.asarray([1.0, 0.0]),
            contact_mode="strike",
            local_lateral_sign=1.0,
            config=config,
            contact_recent=True,
            strike_progress=progress,
        )

    # Out-of-range start requires no fake dynamics; target is still computed.
    assert effect(2.0, 0.0).ankle_target_m[0] == pytest.approx(2 - config.precontact_depth_m)
    end = effect(0.22, 1.0)
    assert not end.active
    assert np.all(end.torque_nm == 0)


@pytest.mark.parametrize("enabled", [False, True])
def test_bilateral_option_requires_opt_in_and_aims_at_opposing_goal(enabled: bool) -> None:
    import math
    from types import SimpleNamespace

    from rosclaw_soccer.growth.locomotion_contact_teacher import G1RollingOptionBridgeConfig
    from rosclaw_soccer.growth.role_self_model import TacticalIntent
    from rosclaw_soccer.skills.team.independent_team_world import _activate_rolling_option
    from rosclaw_soccer.world.field import G1TrainingGoalSpec

    controller = SimpleNamespace(
        spec=SimpleNamespace(yaw_rad=math.pi),
        cell=SimpleNamespace(agent_id="blue.finisher", self_model=SimpleNamespace(team_id="blue")),
        decision=SimpleNamespace(intent=TacticalIntent.SHOOT, target_position_m=(-1.5, 0.0, 1.0)),
        option_active=False,
        option_completed=False,
        qpos_base=0,
        kick_policy=SimpleNamespace(enter=lambda: None, WARMUP_STEPS=50),
        kick_output=SimpleNamespace(),
    )
    qpos = np.asarray([0.55, 0.0, 0.75, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.115])
    _activate_rolling_option(
        controllers=(controller,),
        current_possession_agent_id="blue.finisher",
        last_ball_contact_agent_id="blue.finisher",
        strike_lease_agent_id="blue.finisher",
        frame=1,
        data=SimpleNamespace(qpos=qpos),
        ball_qpos=7,
        goal=replace(G1TrainingGoalSpec(), target_y_m=0.0),
        config=G1RollingOptionBridgeConfig(bilateral_enabled=enabled),
        left_goal_plane_x_m=-1.5,
    )
    assert controller.option_active is enabled
    if enabled:
        assert controller.kick_policy.target_pos_w[0] == -1.5


def test_pending_diagonal_receiver_keeps_negotiated_point() -> None:
    from types import SimpleNamespace

    from rosclaw_soccer.growth.role_self_model import TacticalIntent
    from rosclaw_soccer.skills.team.independent_team_world import (
        IndependentTeamWorldConfig,
        _movement_command,
    )

    controller = SimpleNamespace(
        cell=SimpleNamespace(
            agent_id="red.finisher",
            self_model=SimpleNamespace(
                team_id="red", opponent_ids=(), teammate_ids=(), primary_role="finisher"
            ),
        ),
        qpos_base=0,
        left_ankle_body=0,
        right_ankle_body=1,
        last_world_command=np.zeros(3),
    )
    data = SimpleNamespace(
        qpos=np.asarray([0.0, 0.0, 0.75, 1.0, 0.0, 0.0, 0.0, 0.0, -2.0, 0.115]),
        qvel=np.zeros(12),
        xpos=np.asarray([[0.0, 0.1, 0.03], [0.0, -0.1, 0.03]]),
    )
    decision = SimpleNamespace(intent=TacticalIntent.SUPPORT, target_position_m=(0.0, 0.0, 0.0))
    kwargs = dict(
        controller=controller,
        decision=decision,
        positions={"red.finisher": np.zeros(2)},
        data=data,
        ball_qpos=7,
        ball_qvel=6,
        possession_agent_id=None,
        committed_receiver=True,
        active_receiver=False,
        post_receive_hold=False,
        receive_foot_lateral_offset_m=0.18,
        strike_target_position_m=None,
        config=IndependentTeamWorldConfig(),
    )
    legacy = _movement_command(**kwargs)
    reserved = _movement_command(**kwargs, pending_receive_target_m=(0.0, 0.0))
    assert legacy[1] < 0
    np.testing.assert_array_equal(reserved[:2], [0.0, 0.0])
