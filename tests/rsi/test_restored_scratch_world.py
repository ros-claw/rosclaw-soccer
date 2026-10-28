import mujoco
import numpy as np
import pytest
from test_shared_native_motor_world import world

from rosclaw_soccer.rsi.restored_scratch_world import restored_scratch_world


def test_new_scratch_world_continues_same_native_physics_without_touching_source():
    original = world()
    first = {p: np.full(29, 0.02, dtype=np.float64) for p in original.layouts}
    original.advance(first)
    spec = mujoco.mjtState.mjSTATE_INTEGRATION
    state = np.empty(mujoco.mj_stateSize(original.model, spec), dtype=np.float64)
    mujoco.mj_getState(original.model, original.data, state, spec)
    scratch = restored_scratch_world(
        original.model,
        state,
        players=original.parameters,
        ball_body_name="ball",
        control=1,
        prior_targets=first,
    )
    assert scratch.data is not original.data
    assert scratch.control_steps == 1 and scratch.native_steps == 10
    for p in first:
        np.testing.assert_array_equal(scratch.last_applied_target(p), first[p])
    nxt = {p: np.full(29, -0.01, dtype=np.float64) for p in original.layouts}
    mujoco.mj_forward(original.model, original.data)
    expected = original.advance(nxt)
    actual = scratch.advance(nxt)
    for name in ("qpos", "qvel", "torque", "ball_forces"):
        np.testing.assert_array_equal(actual[name], expected[name])


def test_scratch_restore_rejects_wrong_clock_or_missing_owner():
    original = world()
    targets = {p: np.zeros(29, np.float64) for p in original.layouts}
    original.advance(targets)
    state = np.empty(mujoco.mj_stateSize(original.model, mujoco.mjtState.mjSTATE_INTEGRATION))
    mujoco.mj_getState(original.model, original.data, state, mujoco.mjtState.mjSTATE_INTEGRATION)
    with pytest.raises(ValueError, match="physical time"):
        restored_scratch_world(
            original.model,
            state,
            players=original.parameters,
            ball_body_name="ball",
            control=2,
            prior_targets=targets,
        )
    with pytest.raises(ValueError, match="prior targets"):
        restored_scratch_world(
            original.model,
            state,
            players=original.parameters,
            ball_body_name="ball",
            control=1,
            prior_targets={},
        )


def test_scratch_preserves_float32_neural_target_history():
    original = world()
    targets = {p: np.full(29, 0.02, dtype=np.float32) for p in original.layouts}
    original.advance(targets)
    spec = mujoco.mjtState.mjSTATE_INTEGRATION
    state = np.empty(mujoco.mj_stateSize(original.model, spec), dtype=np.float64)
    mujoco.mj_getState(original.model, original.data, state, spec)
    scratch = restored_scratch_world(
        original.model,
        state,
        players=original.parameters,
        ball_body_name="ball",
        control=1,
        prior_targets=targets,
    )
    for prefix in targets:
        assert scratch.last_applied_target(prefix).dtype == np.float32
        np.testing.assert_array_equal(scratch.last_applied_target(prefix), targets[prefix])
