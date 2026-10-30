import math

import pytest

from rosclaw_soccer.world.match_boundary import require_in_play_ball_launch


def check(position, **kwargs):
    return require_in_play_ball_launch(position, left_x=-1.5, right_x=7.5, radius=0.11, **kwargs)


@pytest.mark.parametrize("position", [(3.0, 0.0, 0.11), (-1.39, 2.89, 0.11), (7.39, -2.89, 1.0)])
def test_whole_ball_inside(position):
    assert check(position) is None


@pytest.mark.parametrize(
    "position",
    [
        (-2.8, 1.3, 0.11),
        (-2.75, 1.2, 0.11),
        (-1.4, 0, 0.11),
        (7.4, 0, 0.11),
        (3, 2.9, 0.11),
        (3, 0, 0.10),
        (math.nan, 0, 0.11),
        (3, math.inf, 0.11),
        (3, 0),
    ],
)
def test_outside_or_invalid_launch_is_rejected(position):
    with pytest.raises(ValueError):
        check(position)


@pytest.mark.parametrize(
    "kwargs", [dict(radius=0), dict(radius=math.nan), dict(left_x=8), dict(half_width=0.1)]
)
def test_invalid_geometry_is_rejected(kwargs):
    geometry = dict(left_x=-1.5, right_x=7.5, radius=0.11)
    geometry.update(kwargs)
    with pytest.raises(ValueError):
        require_in_play_ball_launch((3, 0, 0.11), **geometry)
