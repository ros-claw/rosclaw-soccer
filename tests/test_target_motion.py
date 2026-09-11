"""Target-frame measurements must not reward sideways speed or a turning ray."""

import numpy as np
import pytest

from rosclaw_soccer.sim.target_motion import measure_target_motion


def inputs():
    return [
        np.array([[0.0, 0.0]]),
        np.array([[3.0, 4.0]]),
        np.array([[2.0, 4.0]]),
        np.array([[1.0, 0.0]]),
    ]


def test_speed_and_increment_use_target_not_norm():
    result = measure_target_motion(*inputs())
    assert result.directed_speed_mps[0] == 3.0
    assert result.directed_increment_mps[0] == 1.0
    assert result.direction_resolved[0]


def test_rotating_ray_does_not_create_velocity_increment():
    args = inputs()
    args[2] = args[1].copy()
    args[3] = np.array([[-4.0, 3.0]])
    result = measure_target_motion(*args)
    assert abs(result.directed_speed_mps[0]) < 1e-14
    assert result.directed_increment_mps[0] == 0.0


def test_rotation_translation_invariance():
    args = inputs()
    rotation = np.array([[0.0, -1.0], [1.0, 0.0]])
    transformed = [a @ rotation for a in args]
    transformed[0] += [17.0, -11.0]
    transformed[3] += [17.0, -11.0]
    first, second = measure_target_motion(*args), measure_target_motion(*transformed)
    np.testing.assert_array_equal(first.directed_speed_mps, second.directed_speed_mps)
    np.testing.assert_array_equal(first.directed_increment_mps, second.directed_increment_mps)


@pytest.mark.parametrize("distance", [0.0, 0.005, 0.01])
def test_near_target_is_explicitly_unresolved(distance):
    args = inputs()
    args[3] = np.array([[distance, 0.0]])
    result = measure_target_motion(*args)
    assert result.directed_speed_mps[0] == 3.0 * min(distance / 0.01, 1.0)
    assert bool(result.direction_resolved[0]) == (distance >= 0.01)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf, 1e7])
@pytest.mark.parametrize("index", range(4))
def test_each_input_is_validated(bad, index):
    args = inputs()
    args[index][0, 0] = bad
    with pytest.raises(ValueError):
        measure_target_motion(*args)


@pytest.mark.parametrize(
    "bad",
    [
        np.zeros(2),
        np.zeros((0, 2)),
        np.zeros((4097, 2)),
        np.zeros((1, 3)),
        np.zeros((1, 2), dtype=int),
        [[0.0, 0.0]],
        np.zeros((1, 2), dtype=np.float16),
    ],
)
def test_invalid_shape_or_dtype(bad):
    args = inputs()
    args[0] = bad
    with pytest.raises(ValueError):
        measure_target_motion(*args)


def test_no_broadcasting_or_mutation_and_readonly_independent_results():
    args = inputs()
    copies = [a.copy() for a in args]
    result = measure_target_motion(*args)
    for actual, expected in zip(args, copies, strict=True):
        np.testing.assert_array_equal(actual, expected)
    for output in (
        result.distance_m,
        result.directed_speed_mps,
        result.directed_increment_mps,
        result.direction_resolved,
    ):
        assert not output.flags.writeable
        assert all(not np.shares_memory(output, a) for a in args)
    args[1] = np.repeat(args[1], 2, axis=0)
    with pytest.raises(ValueError):
        measure_target_motion(*args)


def test_float32_and_negative_speed():
    args = [a.astype(np.float32) for a in inputs()]
    args[3] *= -1
    result = measure_target_motion(*args)
    assert result.directed_speed_mps.dtype == np.float64
    assert result.directed_speed_mps[0] == -3.0
    assert result.directed_increment_mps[0] == -1.0
