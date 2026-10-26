import numpy as np
import pytest

from rosclaw_soccer.rsi.cftrack_guarded_reference_aim import select_guarded_aim


def choose(velocity, offset=-0.4):
    return select_guarded_aim(
        ball=np.array([1.0, 0.0, 0.11]),
        velocity=np.array(velocity),
        goal=np.array([5.0, 0.0, 0.11]),
        base=np.array([0.0, 0.0, 0.77]),
        quaternion=np.array([1.0, 0.0, 0.0, 0.0]),
        requested_offset_y_m=offset,
    )


def test_left_uses_original_but_right_retains_requested_aim():
    left = choose((0.0, 0.1, 0.0))
    right = choose((0.0, -0.1, 0.0))
    assert left["fallback_to_original"] and left["selected_offset_y_m"] == 0
    assert not right["fallback_to_original"] and right["selected_offset_y_m"] == -0.4
    assert not left["requested_geometry"]["lateral_within_guard"]
    assert left["original_geometry"]["lateral_within_guard"]


def test_when_both_options_violate_guard_selection_abstains():
    with pytest.raises(ValueError, match="neither requested nor original"):
        choose((0.0, 0.14, 0.0))


@pytest.mark.parametrize("offset", [float("nan"), 0.71, True])
def test_invalid_aim_does_not_reach_planning(offset):
    with pytest.raises(ValueError):
        choose((0.0, 0.0, 0.0), offset)
