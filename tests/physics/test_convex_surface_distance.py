import itertools

import numpy as np
import pytest

from rosclaw_soccer.physics.convex_surface_distance import ConvexSurfaceDistance


def cube():
    triangles, planes = [], []
    for axis, side in itertools.product(range(3), (-1, 1)):
        other = [i for i in range(3) if i != axis]
        corners = []
        for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
            p = np.zeros(3)
            p[axis] = side
            p[other] = [a, b]
            corners.append(p)
        triangles.extend([[corners[i] for i in face] for face in ((0, 1, 2), (0, 2, 3))])
        plane = np.zeros(4)
        plane[axis] = side
        plane[3] = -1
        planes.append(plane)
    return ConvexSurfaceDistance(np.asarray(triangles), np.asarray(planes))


@pytest.mark.parametrize(
    "point,expected,closest",
    [
        ([2, 0, 0], 1, [1, 0, 0]),
        ([2, 2, 0], np.sqrt(2), [1, 1, 0]),
        ([2, 2, 2], np.sqrt(3), [1, 1, 1]),
        ([0, 0, 0], -1, None),
        ([1, 0, 0], 0, [1, 0, 0]),
    ],
)
def test_face_edge_vertex_inside_and_boundary(point, expected, closest):
    distance, actual = cube().nearest(np.asarray(point, dtype=np.float64))
    assert distance == pytest.approx(expected, abs=1e-12)
    if closest is not None:
        np.testing.assert_allclose(actual, closest, atol=1e-12)


def test_reject_nonfinite_query():
    with pytest.raises(ValueError):
        cube().nearest(np.array([np.nan, 0.0, 0.0]))


def test_mutable_caller_triangle_does_not_alias_surface():
    triangles = np.array(
        [
            [[0, 0, 0], [1, 0, 0], [0, 1, 0]],
            [[0, 0, 0], [1, 0, 0], [0, 0, 1]],
            [[0, 0, 0], [0, 1, 0], [0, 0, 1]],
            [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        ],
        dtype=np.float64,
    )
    planes = np.array(
        [[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [1, 1, 1, -1]], dtype=np.float64
    )
    planes[-1] /= np.sqrt(3)
    surface = ConvexSurfaceDistance(triangles, planes)
    expected = surface.nearest(np.array([2.0, 0.0, 0.0]))
    triangles[:] = 100
    planes[:] = 0
    actual = surface.nearest(np.array([2.0, 0.0, 0.0]))
    assert actual[0] == expected[0]
    np.testing.assert_array_equal(actual[1], expected[1])
