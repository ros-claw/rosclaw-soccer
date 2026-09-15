import pytest

from rosclaw_soccer.growth.motor_option_lifecycle import MotorOptionLifecycle


def ready(state, frame, distance=0.7, body=True, receive=False):
    return state.ready(frame, ball_distance_m=distance, body_ready=body, incoming_receive=receive)


def test_successful_contact_requires_a_new_encounter_and_recovery():
    state = MotorOptionLifecycle()
    assert not ready(state, 0)
    state.start(10)
    state.finish(100, contact_observed=True)
    assert not ready(state, 1000)
    assert not ready(state, 1001, distance=2.0)
    assert not ready(state, 1002, body=False)
    assert not ready(state, 1003, receive=True)
    assert ready(state, 1004)
    state.start(1004)
    assert state.activation_count == 2 and not ready(state, 1005)


def test_missed_option_has_two_bounded_retries_not_permanent_lock_or_spam():
    state = MotorOptionLifecycle()
    for attempt in range(3):
        start = attempt * 250
        state.start(start)
        state.finish(start + 100, contact_observed=False)
        assert not ready(state, start + 199)
        assert ready(state, start + 200) == (attempt < 2)
    assert not ready(state, 10000)
    assert not ready(state, 10001, distance=2.0)
    assert ready(state, 10002)
    state.start(10002)
    assert state.consecutive_misses == 0


def test_rearm_never_happens_during_an_option_and_rejects_bad_time():
    state = MotorOptionLifecycle()
    with pytest.raises(ValueError):
        state.finish(1, contact_observed=True)
    state.start(1)
    with pytest.raises(ValueError):
        state.start(2)
    assert not ready(state, 1000)
    state.finish(1001, contact_observed=True)
    with pytest.raises(ValueError):
        ready(state, 1)
    with pytest.raises(ValueError):
        ready(state, 1002, distance=float("nan"))


@pytest.mark.parametrize("enabled,stance_ready", [(False, True), (True, True), (True, False)])
def test_world_rearm_still_requires_existing_motor_stance_admission(enabled, stance_ready):
    from test_motor_task_context import controller_fixture

    from rosclaw_soccer.growth.locomotion_contact_teacher import G1RollingOptionBridgeConfig
    from rosclaw_soccer.skills.team.independent_team_world import _activate_rolling_option
    from rosclaw_soccer.world.field import G1TrainingGoalSpec

    c, data = controller_fixture(y=0.0)
    c.option_completed = True
    c.option_lifecycle = MotorOptionLifecycle()
    c.option_lifecycle.start(0)
    c.option_lifecycle.finish(50, contact_observed=False)
    c.option_rearm_ready = ready(c.option_lifecycle, 200)
    if not stance_ready:
        data.qpos[0] = 3.2
    _activate_rolling_option(
        controllers=(c,),
        current_possession_agent_id=c.cell.agent_id,
        last_ball_contact_agent_id=c.cell.agent_id,
        strike_lease_agent_id=None,
        frame=200,
        data=data,
        ball_qpos=7,
        goal=G1TrainingGoalSpec(),
        config=G1RollingOptionBridgeConfig(continuous_rearm_enabled=enabled),
    )
    assert c.option_active is (enabled and stance_ready)
    assert c.option_lifecycle.activation_count == (2 if enabled and stance_ready else 1)


def test_default_option_hash_preserved_and_flag_is_not_numeric():
    from dataclasses import asdict

    from rosclaw_soccer.growth.locomotion_contact_teacher import G1RollingOptionBridgeConfig
    from rosclaw_soccer.sim.contracts import hash_json

    config = G1RollingOptionBridgeConfig()
    legacy = asdict(config)
    legacy.pop("continuous_rearm_enabled")
    legacy.pop("task_context_bound")
    assert config.config_hash == hash_json(legacy)
    with pytest.raises(ValueError):
        G1RollingOptionBridgeConfig(continuous_rearm_enabled=1)
