import numpy as np
import pytest

from rosclaw_soccer.rsi.live_neural_transition import blend_live_targets, quintic_weight
from rosclaw_soccer.rsi.paid_velocity_shadow import policy_action_from_hg_target
from rosclaw_soccer.rsi.shared_native_motor_world import NativePDParameters


def pd(stiffness=20.0, effort=25.0):
    return NativePDParameters(np.full(29, stiffness), np.ones(29), np.full(29, effort))


def test_schedule_is_monotone_with_exact_endpoints():
    values = [quintic_weight(i, start=285, duration=20) for i in range(320)]
    assert values[284] == values[285] == 0
    assert values[305] == values[306] == 1
    assert values[295] == 0.5
    assert all(x <= y for x, y in zip(values, values[1:], strict=False))


@pytest.mark.parametrize(
    "values", [(0, 0, 0), (0, 0, 101), (0, -1, 20), (-1, 0, 20), (True, 0, 20)]
)
def test_schedule_rejects_ambiguous_or_unbounded_clock(values):
    with pytest.raises(ValueError):
        quintic_weight(values[0], start=values[1], duration=values[2])


def test_endpoints_are_exact_and_do_not_alias_live_targets_or_profiles():
    a, b = np.linspace(-0.3, 0.3, 29), np.linspace(0.1, 0.4, 29)
    p, q = pd(), pd(60.0)
    for weight, expected in ((0.0, a), (1.0, b)):
        target, profile = blend_live_targets(a, b, p, q, weight=weight)
        np.testing.assert_array_equal(target, expected)
        target[0] += 1
        assert a[0] == -0.3 and b[0] == 0.1
        assert not profile.effort.flags.writeable
    target, profile = blend_live_targets(a, b, p, q, weight=0.5)
    np.testing.assert_array_equal(target, 0.5 * a + 0.5 * b)
    np.testing.assert_array_equal(profile.stiffness, np.full(29, 40.0))


def test_blend_depends_on_both_current_proposals_not_last_target():
    a, b = np.zeros(29), np.ones(29)
    first, _ = blend_live_targets(a, b, pd(), pd(), weight=0.5)
    b[4] = 2.0
    second, _ = blend_live_targets(a, b, pd(), pd(), weight=0.5)
    assert second[4] - first[4] == 0.5


@pytest.mark.parametrize("weight", [-0.1, 1.1, float("nan"), float("inf")])
def test_invalid_weight_rejected(weight):
    with pytest.raises(ValueError):
        blend_live_targets(np.zeros(29), np.ones(29), pd(), pd(), weight=weight)


def test_nonfinite_unused_endpoint_and_changed_effort_still_rejected():
    bad = np.ones(29)
    bad[0] = np.nan
    with pytest.raises(ValueError):
        blend_live_targets(np.zeros(29), bad, pd(), pd(), weight=0.0)
    with pytest.raises(ValueError):
        blend_live_targets(np.zeros(29), np.ones(29), pd(), pd(effort=26.0), weight=0.5)


def test_executed_target_maps_to_both_private_action_orders():
    target, _ = blend_live_targets(np.zeros(29), np.linspace(-0.3, 0.3, 29), pd(), pd(), weight=0.5)
    # This test intentionally uses a different nontrivial permutation for the
    # second policy: HG action indices cannot be mistaken for Isaac indices.
    order = np.roll(np.arange(29), 7)
    offsets, scales = np.linspace(-0.1, 0.1, 29), np.linspace(0.1, 0.5, 29)
    encoded = policy_action_from_hg_target(target, offsets, scales, order)
    np.testing.assert_allclose(offsets + encoded[order] * scales, target, atol=3e-7, rtol=0)
    assert not np.array_equal(
        encoded, policy_action_from_hg_target(target, offsets, scales, np.arange(29))
    )
