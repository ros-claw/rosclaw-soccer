import numpy as np
import pytest

from rosclaw_soccer.rsi.causal_reference_bridge import CausalReferenceBridge


def test_measured_start_and_exact_original_reference_after_declared_bridge():
    q = np.linspace(-0.2, 0.2, 29).astype(np.float32)
    bridge = CausalReferenceBridge(q, controls=15)
    original = np.full(29, 0.5, np.float32)
    v, w = np.ones(29, np.float32), np.ones(3, np.float32)
    start = bridge.reference(original, v, w, frame=0)
    np.testing.assert_array_equal(start[0], q)
    assert not start[1].any() and not start[2].any()
    q[:] = 100  # input ownership
    assert bridge.initial.max() < 1
    for age in (15, 16, 100):
        for actual, expected in zip(
            bridge.reference(original, v, w, frame=age), (original, v, w), strict=True
        ):
            np.testing.assert_array_equal(actual, expected)


def test_velocity_is_derivative_of_declared_pose_curve_not_unrelated_reference():
    bridge = CausalReferenceBridge(np.zeros(29, np.float32), controls=20)
    pos, vel, ang = np.ones(29, np.float32), np.zeros(29, np.float32), np.zeros(3, np.float32)
    q, v, _ = bridge.reference(pos, vel, ang, frame=10)
    np.testing.assert_allclose(q, 0.5)
    np.testing.assert_allclose(v, 3.75)


@pytest.mark.parametrize("count", [True, 0, 51, -1])
def test_invalid_horizon_rejected(count):
    with pytest.raises(ValueError):
        CausalReferenceBridge(np.zeros(29, np.float32), controls=count)


def test_player_bridge_changes_only_reference_observation_not_actual_state_or_targets():
    from test_shared_native_neural_player import players

    from rosclaw_soccer.rsi.paid_skill_handoff import snapshot
    from rosclaw_soccer.rsi.shared_native_neural_player import SharedNativeNeuralPlayer

    world, templates = players()
    template = templates["b/"]
    hg = np.linspace(-0.1, 0.1, 29)
    world.data.qpos[world.layouts["b/"].qi] = hg  # declared initial fixture, no live step
    before = snapshot(world.data)
    order = np.arange(28, -1, -1)

    def observe(view, motion, frame, previous, ref_q, ref_v, ref_w, *args, **kwargs):
        obs = np.zeros(160, np.float32)
        obs[:29] = ref_q
        obs[29:58] = ref_v
        return obs

    player = SharedNativeNeuralPlayer(
        world,
        "b/",
        template.actor,
        clip=template.clip,
        torso_index=0,
        build_observation=observe,
        goal=template.goal,
        offset=template.offset,
        scale=template.scale,
        order=order,
        reference_bridge_controls=15,
    )
    target, trace = player.propose()
    np.testing.assert_array_equal(trace["NN_obs"][:29], hg[order].astype(np.float32))
    np.testing.assert_array_equal(
        target, template.offset + trace["NN_action"][order] * template.scale
    )
    for key, value in before.items():
        np.testing.assert_array_equal(value, snapshot(world.data)[key])
