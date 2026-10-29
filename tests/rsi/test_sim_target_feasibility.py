import mujoco
import numpy as np
import pytest
from test_shared_cftrack_forecast import Policy, setup

from rosclaw_soccer.rsi.shared_native_motor_world import SharedNativeMotorWorld
from rosclaw_soccer.rsi.sim_target_feasibility import (
    ProjectedSharedOmniContactPlayer,
    TargetFeasibility,
)


def test_range_and_rate_and_commit_are_causal():
    filt = TargetFeasibility(np.zeros(29), np.tile([-0.3, 0.3], (29, 1)))
    first = filt.propose(np.full(29, 10.0))
    np.testing.assert_array_equal(first, np.full(29, 0.08))
    with pytest.raises(ValueError):
        filt.propose(np.zeros(29))
    with pytest.raises(ValueError):
        filt.commit(np.zeros(29))
    filt.commit(first)
    second = filt.propose(np.full(29, 10.0))
    np.testing.assert_array_equal(second, np.full(29, 0.16))
    filt.commit(second)
    third = filt.propose(np.full(29, -10.0))
    np.testing.assert_array_equal(third, np.full(29, 0.08))


def test_initial_state_is_never_silently_projected():
    limits = np.tile([-0.3, 0.3], (29, 1))
    initial = np.zeros(29)
    initial[3] = 0.31
    with pytest.raises(ValueError):
        TargetFeasibility(initial, limits)


def test_nonfinite_and_bad_limits_rejected():
    limits = np.tile([-0.3, 0.3], (29, 1))
    filt = TargetFeasibility(np.zeros(29), limits)
    x = np.zeros(29)
    x[0] = np.nan
    with pytest.raises(ValueError):
        filt.propose(x)
    limits[1, 0] = np.inf
    with pytest.raises(ValueError):
        TargetFeasibility(np.zeros(29), limits)


def test_actual_sim_commit_uses_projected_targets(tmp_path):
    original, _ = setup(tmp_path, physical_range=0.3, cold=True)
    world0 = original.world
    data = mujoco.MjData(world0.model)
    mujoco.mj_copyData(data, world0.model, world0.data)
    world = SharedNativeMotorWorld(
        world0.model, data, players=world0.parameters, ball_body_name="ball"
    )
    actor = ProjectedSharedOmniContactPlayer(
        world,
        "a/",
        Policy(original.policy.ort_session),
        expected_policy_type=Policy,
        goal=np.array([5.0, 0.0, 0.11]),
    )
    for _ in range(3):
        target, measurement = actor.propose()
        assert np.max(np.abs(target)) <= 0.3
        np.testing.assert_array_equal(
            target, measurement["nn_decoded_target"] + measurement["target_projection"]
        )
        world.advance({"a/": target})
        actor.commit()
    assert world.control_steps == 3 and not actor.failed
