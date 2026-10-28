import mujoco
import numpy as np
import pytest
from test_shared_native_neural_player import players

from rosclaw_soccer.rsi.restored_scratch_world import restored_scratch_world
from rosclaw_soccer.rsi.restored_shadow_motor import restored_shadow_motor
from rosclaw_soccer.rsi.shared_shadow_motor import SharedShadowMotor


def test_restored_own_shadow_observation_and_actual_handoff_match():
    source, players_by_prefix = players()
    template = players_by_prefix["b/"]
    args = dict(
        clip=template.clip,
        torso_index=template.torso_index,
        build_observation=template.build_observation,
        goal=template.goal,
        offset=template.offset,
        scale=template.scale,
        order=template.order,
    )
    shadow = SharedShadowMotor(source, "b/", template.actor, **args)
    targets = {"a/": np.zeros(29), "b/": np.full(29, 0.08)}
    for _ in range(2):
        mujoco.mj_forward(source.model, source.data)
        shadow.observe()
        source.advance(targets)
        shadow.commit_actual()
    spec = mujoco.mjtState.mjSTATE_INTEGRATION
    state = np.empty(mujoco.mj_stateSize(source.model, spec), dtype=np.float64)
    mujoco.mj_getState(source.model, source.data, state, spec)
    scratch = restored_scratch_world(
        source.model,
        state,
        players=source.parameters,
        ball_body_name="ball",
        control=2,
        prior_targets=targets,
    )
    restored = restored_shadow_motor(
        scratch,
        "b/",
        template.actor,
        **args,
        previous_action=shadow.previous_action.copy(),
        hidden=shadow.state[0].numpy().copy(),
        cell=shadow.state[1].numpy().copy(),
        observed_controls=2,
    )
    mujoco.mj_forward(source.model, source.data)
    expected, actual = shadow.observe(), restored.observe()
    for key in expected:
        np.testing.assert_array_equal(expected[key], actual[key])
    expected_physics, actual_physics = source.advance(targets), scratch.advance(targets)
    for key in ("qpos", "qvel", "torque", "ball_forces"):
        np.testing.assert_array_equal(expected_physics[key], actual_physics[key])
    shadow.commit_actual()
    restored.commit_actual()
    warm = restored.handoff(goal=template.goal, carry_memory=True)
    np.testing.assert_array_equal(warm.last_action, shadow.previous_action)
    np.testing.assert_array_equal(warm.state[0].numpy(), shadow.state[0].numpy())


def test_restored_shadow_rejects_unmatched_action_or_control():
    source, ps = players()
    template = ps["b/"]
    source.advance({"a/": np.zeros(29), "b/": np.full(29, 0.08)})
    args = dict(
        clip=template.clip,
        torso_index=template.torso_index,
        build_observation=template.build_observation,
        goal=template.goal,
        offset=template.offset,
        scale=template.scale,
        order=template.order,
        previous_action=np.zeros(29, np.float32),
        hidden=np.zeros((2, 1, 256), np.float32),
        cell=np.zeros((2, 1, 256), np.float32),
        observed_controls=1,
    )
    with pytest.raises(ValueError, match="actual last target"):
        restored_shadow_motor(source, "b/", template.actor, **args)
