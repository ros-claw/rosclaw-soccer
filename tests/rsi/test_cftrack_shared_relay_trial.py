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
