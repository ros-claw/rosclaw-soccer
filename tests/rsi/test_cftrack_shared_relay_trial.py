from types import SimpleNamespace

import numpy as np
import pytest

from rosclaw_soccer.rsi.cftrack_shared_relay_trial import (
    early_incoming_boundary,
    run_shared_cftrack_trial,
)


@pytest.mark.parametrize("context,side,arm", [(2, 0.0, True), (2, 0.35, 1), (True, 0.35, False)])
def test_unregistered_course_fails_before_allocating_world(context, side, arm):
    class ForbiddenSetup:
        def prepare(self, *args):
            raise AssertionError("invalid declaration allocated physics")

    with pytest.raises(ValueError):
        run_shared_cftrack_trial(
            ForbiddenSetup(),
            object(),
            context=context,
            lateral_m=side,
            use_cftrack=arm,
            policy_factory=lambda: None,
            expected_policy_type=object,
        )


def test_early_incoming_uses_current_world_and_requires_five_boundaries():
    velocity = np.zeros(41)
    velocity[35:37] = [0.6, -0.2]
    layout = SimpleNamespace(canonical=lambda data: (None, data.velocity))
    world = SimpleNamespace(
        layouts={"b/": layout},
        data=SimpleNamespace(velocity=velocity),
        control_steps=4,
        failed=False,
        _require_current_boundary=lambda: None,
    )
    before = velocity.copy()
    assert early_incoming_boundary(world, "b/", np.array([1.0, 0.0])) == (False, 0.6)
    world.control_steps = 5
    assert early_incoming_boundary(world, "b/", np.array([1.0, 0.0])) == (True, 0.6)
    world.failed = True
    assert early_incoming_boundary(world, "b/", np.array([1.0, 0.0])) == (False, 0.6)
    assert np.array_equal(velocity, before)
    with pytest.raises(ValueError):
        early_incoming_boundary(world, "b/", np.array([2.0, 0.0]))
    velocity[35] = np.nan
    with pytest.raises(ValueError):
        early_incoming_boundary(world, "b/", np.array([1.0, 0.0]))


def test_explicit_receiver_actor_cannot_be_mixed_with_cftrack():
    class ForbiddenSetup:
        def prepare(self, *args):
            raise AssertionError("invalid declaration allocated physics")

    with pytest.raises(ValueError):
        run_shared_cftrack_trial(
            ForbiddenSetup(),
            object(),
            context=2,
            lateral_m=0.35,
            use_cftrack=True,
            receiver_actor=object(),
            policy_factory=lambda: None,
            expected_policy_type=object,
        )


def test_passer_recovery_must_be_explicit_boolean_before_world_allocation():
    class ForbiddenSetup:
        def prepare(self, *args):
            raise AssertionError("invalid declaration allocated physics")

    with pytest.raises(ValueError):
        run_shared_cftrack_trial(
            ForbiddenSetup(),
            object(),
            context=2,
            lateral_m=0.35,
            use_cftrack=False,
            passer_velocity_recovery=1,
            policy_factory=lambda: None,
            expected_policy_type=object,
        )


@pytest.mark.parametrize("offset", [np.array([np.nan, 0.0]), np.array([0.06, 0.0]), [0.02, 0.0]])
def test_invalid_ball_offset_fails_before_world_allocation(offset):
    class ForbiddenSetup:
        def prepare(self, *args, **kwargs):
            raise AssertionError("invalid offset allocated physics")

    with pytest.raises(ValueError):
        run_shared_cftrack_trial(
            ForbiddenSetup(),
            object(),
            context=2,
            lateral_m=0.35,
            use_cftrack=False,
            policy_factory=lambda: None,
            expected_policy_type=object,
            ball_planar_offset_world=offset,
        )


def test_ball_offset_is_forwarded_only_to_initial_setup():
    class Prepared(Exception):
        pass

    class InspectSetup:
        def prepare(self, context, side, *, ball_planar_offset_world):
            assert (context, side) == (2, 0.35)
            np.testing.assert_array_equal(ball_planar_offset_world, [0.02, 0.0])
            raise Prepared

    with pytest.raises(Prepared):
        run_shared_cftrack_trial(
            InspectSetup(),
            object(),
            context=2,
            lateral_m=0.35,
            use_cftrack=False,
            policy_factory=lambda: None,
            expected_policy_type=object,
            ball_planar_offset_world=np.array([0.02, 0.0]),
        )


@pytest.mark.parametrize("rate", [0, 0.49, 1.01, float("nan"), float("inf"), True, "0.75"])
def test_invalid_passer_reference_rate_fails_before_world_allocation(rate):
    class ForbiddenSetup:
        def prepare(self, *args, **kwargs):
            raise AssertionError("invalid rate allocated physics")

    with pytest.raises(ValueError):
        run_shared_cftrack_trial(
            ForbiddenSetup(),
            object(),
            context=2,
            lateral_m=0.35,
            use_cftrack=False,
            policy_factory=lambda: None,
            expected_policy_type=object,
            passer_reference_rate=rate,
        )


def test_passer_reference_rate_targets_only_the_first_player(monkeypatch):
    from rosclaw_soccer.rsi import cftrack_shared_relay_trial as module

    class Observed(Exception):
        pass

    world = SimpleNamespace(data=SimpleNamespace(qpos=np.zeros(3), qvel=np.zeros(3)))
    episode = SimpleNamespace(
        clip=object(),
        torso_index=0,
        build_observation=object(),
        goal=object(),
        offset=object(),
        scale=object(),
        order=object(),
    )
    setup = SimpleNamespace(prepare=lambda context, side: (episode, world, None, None, None))
    monkeypatch.setattr(module, "motion_segment", lambda clip, length: object())

    def observe(actual_world, prefix, actor, **kwargs):
        assert actual_world is world and prefix == "a/" and kwargs["reference_rate"] == 0.75
        raise Observed

    monkeypatch.setattr(module, "SharedNativeNeuralPlayer", observe)
    with pytest.raises(Observed):
        run_shared_cftrack_trial(
            setup,
            object(),
            context=2,
            lateral_m=0.35,
            use_cftrack=False,
            policy_factory=lambda: None,
            expected_policy_type=object,
            passer_reference_rate=0.75,
        )


def test_historical_baseline_identity_ignores_only_local_index_and_paths():
    from scripts.rsi_run_shared_router_relay import same_baseline_metadata

    previous = dict(
        index=1,
        context=2,
        lateral_m=-0.35,
        arm="router",
        controls=300,
        artifacts={"physics": "old"},
    )
    same = dict(previous, index=0, artifacts={"physics": "new"})
    assert same_baseline_metadata(same, previous)
    assert not same_baseline_metadata(dict(same, context=3), previous)
    assert not same_baseline_metadata(dict(same, controls=299), previous)


def test_passer_recovery_uses_pinned_continuous_600_control_stream(monkeypatch):
    from rosclaw_soccer.rsi import cftrack_shared_relay_trial as module

    calls = []
    stream = object()
    monkeypatch.setattr(
        module,
        "ContinuousVelocityStream",
        lambda model, config, **kwargs: calls.append((model, config, kwargs)) or stream,
    )
    monkeypatch.setattr(module, "PasserVelocityRecovery", lambda w, prefix, s: (w, prefix, s))
    setup = SimpleNamespace(
        velocity_path="model.onnx",
        velocity_config="config.yaml",
        pins={"model.onnx": "model-hash", "config.yaml": "config-hash"},
    )
    world = object()
    assert module.new_passer_recovery(setup, world) == (world, "a/", stream)
    assert calls == [
        (
            "model.onnx",
            "config.yaml",
            {
                "expected_model_hash": "model-hash",
                "expected_config_hash": "config-hash",
                "horizon_controls": 600,
            },
        )
    ]
