import numpy as np
import pytest

from rosclaw_soccer.rsi.cftrack_velocity_intercept import predict_initial_intercept


def test_prediction_uses_current_velocity_without_changing_measured_arrays():
    ball = np.array([1.0, 0.0, 0.11])
    velocity = np.array([0.1, 0.0, 0.0])
    goal = np.array([5.0, 0.0, 0.11])
    expected_ball, expected_velocity = ball.copy(), velocity.copy()
    predicted, receipt = predict_initial_intercept(ball, velocity, goal)
    assert receipt["predicted_horizon_s"] == pytest.approx(1.44 / (1 - 0.1 / 0.8))
    assert predicted[0] > ball[0] and predicted[1] == ball[1]
    assert (
        receipt["prediction_is_not_a_measured_future"] and receipt["privileged_sim_velocity_input"]
    )
    np.testing.assert_array_equal(ball, expected_ball)
    np.testing.assert_array_equal(velocity, expected_velocity)


@pytest.mark.parametrize(
    "velocity",
    [
        [0.1, 0.0, 0.0],
        np.array([0.21, 0.0, 0.0]),
        np.array([0.0, 0.0, 0.1]),
        np.array([float("nan"), 0.0, 0.0]),
    ],
)
def test_invalid_velocity_is_rejected(velocity):
    with pytest.raises(ValueError):
        predict_initial_intercept(np.array([1.0, 0.0, 0.11]), velocity, np.array([5.0, 0.0, 0.11]))


def test_lateral_and_backward_predictions_follow_measured_direction():
    ball = np.array([1.0, 0.0, 0.11])
    goal = np.array([5.0, 0.0, 0.11])
    for vel in (np.array([-0.1, 0.0, 0.0]), np.array([0.0, -0.1, 0.0])):
        predicted, receipt = predict_initial_intercept(ball, vel, goal)
        np.testing.assert_allclose(predicted - ball, vel * receipt["predicted_horizon_s"])


@pytest.mark.parametrize("fail_plan", [False, True])
@pytest.mark.parametrize("aim", [0.0, -0.4])
def test_reference_planning_restores_measured_ball_before_return_or_failure(
    monkeypatch, fail_plan, aim
):
    import sys
    from types import ModuleType, SimpleNamespace

    from rosclaw_soccer.rsi import cftrack_velocity_intercept as module
    from rosclaw_soccer.rsi.shared_omnicontact_player import SharedOmniContactPlayer

    actual = np.array([1.0, 0.0, 0.11])
    state = SimpleNamespace(
        ball_pos=actual.copy(),
        obj_pos=actual.copy(),
        q=np.zeros(29),
        base_pos=np.zeros(3),
        base_quat=np.array([1.0, 0.0, 0.0, 0.0]),
    )
    kin = SimpleNamespace(forward=lambda *args: {"pelvis": {"pos": np.zeros(3)}})
    policy = SimpleNamespace(
        state_cmd=state,
        kinematics=kin,
        counter_step=0,
        task="kickball",
        reference_source="CFgen",
        async_stage_thread=None,
        obs_history_buffer=np.zeros((5, 141)),
        action=np.zeros(29),
        goal_pos=np.array([5.0, 0.0, 0.11]),
    )
    data = SimpleNamespace(
        xpos=np.array([[0.0, 0.0, 0.0], actual]),
        qpos=np.zeros(43),
        qvel=np.array([0.1, 0.0, 0.0] + [0.0] * 38),
        time=0.0,
    )
    world = SimpleNamespace(
        _require_current_boundary=lambda: None,
        data=data,
        native_steps=0,
        control_steps=0,
        _torque_layer_owners={},
    )
    player = object.__new__(SharedOmniContactPlayer)
    player.world, player.policy = world, policy
    player.layout = SimpleNamespace(ball=1, bv=0)
    player.failed, player.pending, player.motor_plan = False, None, None
    pkg = ModuleType("policy")
    pkg.__path__ = []
    sub = ModuleType("policy.omnicontact")
    sub.__path__ = []
    ref = ModuleType("policy.omnicontact.CFgen_reference")

    def generate(p, fk):
        assert p.state_cmd.ball_pos[0] > actual[0]
        assert p.state_cmd.obj_pos[0] > actual[0]
        assert p.goal_pos[1] == aim
        if fail_plan:
            raise RuntimeError("generator failed")

    ref.plan_cfgen_reference = generate
    monkeypatch.setitem(sys.modules, "policy", pkg)
    monkeypatch.setitem(sys.modules, "policy.omnicontact", sub)
    monkeypatch.setitem(sys.modules, "policy.omnicontact.CFgen_reference", ref)
    monkeypatch.setattr(module, "cftrack_state_hash", lambda p: "sha256:" + "a" * 64)
    monkeypatch.setattr(module, "condition_direct_kick_approach", lambda p: {"test": True})
    if fail_plan:
        with pytest.raises(RuntimeError, match="generator failed"):
            module.condition_velocity_intercept(player, reference_goal_offset_y_m=aim)
        assert player.failed
    else:
        receipt = module.condition_velocity_intercept(player, reference_goal_offset_y_m=aim)
        assert receipt["measured_ball_observations_restored_before_inference"]
        assert not player.failed
    np.testing.assert_array_equal(state.ball_pos, actual)
    np.testing.assert_array_equal(state.obj_pos, actual)
    np.testing.assert_array_equal(policy.goal_pos, [5.0, 0.0, 0.11])
    assert data.time == 0.0 and world.native_steps == 0
