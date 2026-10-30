import numpy as np
import pytest

from rosclaw_soccer.rsi.native_receive_bank import heading_front_launch, heading_launch


@pytest.mark.parametrize(
    "quaternion", [np.array([1.0, 0.0, 0.0, 0.0]), np.array([2**-0.5, 0.0, 0.0, 2**-0.5])]
)
def test_team_sides_share_identical_body_front_course(quaternion):
    root = (3.0, -2.0, 0.8)
    red_pos, red_v = (3.9, -1.8, 0.11), (-1.25, 0.0, 0.0)
    blue_pos, blue_v = (2.1, -2.2, 0.11), (1.25, 0.0, 0.0)
    red = heading_front_launch(red_pos, red_v, root, quaternion, team_sign=1.0)
    blue = heading_front_launch(blue_pos, blue_v, root, quaternion, team_sign=-1.0)
    np.testing.assert_allclose(red, blue, rtol=0, atol=1e-15)
    legacy_blue = heading_launch(blue_pos, blue_v, root, quaternion)
    assert not np.allclose(red[0], legacy_blue[0])


def test_invalid_sign_fails_closed():
    with pytest.raises(ValueError):
        heading_front_launch(
            (0.0, 0.0, 0.11),
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            np.array([1.0, 0.0, 0.0, 0.0]),
            team_sign=0.0,
        )
