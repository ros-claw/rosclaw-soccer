"""Reference adaptation never substitutes for measured physical safety."""

import numpy as np
import pytest

from rosclaw_soccer.providers.g1.joint_boundary_guard import (
    project_g1_policy_position_reference,
)


def inputs():
    return dict(
        target_position=np.linspace(-2.0, 2.0, 29),
        joint_ranges=np.tile([-1.0, 1.0], (29, 1)),
        limited=np.ones(29, dtype=bool),
    )


def test_projection_copies_and_matches_clipping():
    values = inputs()
    before = {key: value.copy() for key, value in values.items()}
    result = project_g1_policy_position_reference(**values, margin_rad=0.05)
    np.testing.assert_array_equal(result, np.clip(before["target_position"], -0.95, 0.95))
    assert not np.shares_memory(result, values["target_position"])
    for key in values:
        np.testing.assert_array_equal(values[key], before[key])


def test_unlimited_targets_unchanged_and_degenerate_unlimited_range_allowed():
    values = inputs()
    values["limited"][:] = False
    values["joint_ranges"][:] = 0
    result = project_g1_policy_position_reference(**values)
    np.testing.assert_array_equal(result, values["target_position"])
    assert not np.shares_memory(result, values["target_position"])


def test_readonly_inputs_and_exact_boundary_are_supported():
    values = inputs()
    values["target_position"] = np.clip(values["target_position"], -1, 1)
    for value in values.values():
        value.flags.writeable = False
    result = project_g1_policy_position_reference(**values)
    np.testing.assert_array_equal(result, values["target_position"])


@pytest.mark.parametrize("margin", [True, False, -0.001, 0.101, float("nan"), float("inf"), "0"])
def test_invalid_margin_rejected(margin):
    with pytest.raises(ValueError):
        project_g1_policy_position_reference(**inputs(), margin_rad=margin)


@pytest.mark.parametrize("key", ["target_position", "joint_ranges", "limited"])
def test_bad_shapes_rejected(key):
    values = inputs()
    values[key] = values[key][:-1]
    with pytest.raises(ValueError):
        project_g1_policy_position_reference(**values)


@pytest.mark.parametrize("key", ["target_position", "joint_ranges"])
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_rejected(key, bad):
    values = inputs()
    values[key].flat[0] = bad
    with pytest.raises(ValueError):
        project_g1_policy_position_reference(**values)


@pytest.mark.parametrize("kind", ["object", "complex128", "bool"])
def test_nonreal_or_boolean_targets_rejected(kind):
    values = inputs()
    values["target_position"] = values["target_position"].astype(kind)
    with pytest.raises(ValueError):
        project_g1_policy_position_reference(**values)


def test_numeric_mask_not_silently_coerced():
    values = inputs()
    values["limited"] = values["limited"].astype(int)
    with pytest.raises(ValueError):
        project_g1_policy_position_reference(**values)


@pytest.mark.parametrize("bounds", [[0.0, 0.0], [1.0, -1.0], [-0.1, 0.1]])
def test_collapsed_limited_range_rejected(bounds):
    values = inputs()
    values["joint_ranges"][0] = bounds
    with pytest.raises(ValueError):
        project_g1_policy_position_reference(**values, margin_rad=0.1)


def test_projection_idempotent_and_preserves_unlimited_entries():
    values = inputs()
    values["limited"][::2] = False
    first = project_g1_policy_position_reference(**values, margin_rad=0.1)
    second = project_g1_policy_position_reference(
        **{**values, "target_position": first}, margin_rad=0.1
    )
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(first[::2], values["target_position"][::2])
