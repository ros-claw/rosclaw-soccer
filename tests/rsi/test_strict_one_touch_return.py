import numpy as np

from rosclaw_soccer.rsi.strict_one_touch_return import strict_one_touch_return


def test_direct_return_is_actual_clean_ordered_motion_not_trap():
    ball = np.zeros((20, 3))
    velocity = np.zeros((20, 2))
    force = np.zeros((20, 2, 3))
    failed = np.zeros((2, 2), dtype=np.bool_)
    target = np.array([2.0, 0, 0.11])
    origin = np.array([0.0, 0])
    force[2, 0, 0], force[5, 1, 1] = 11, 11
    velocity[5] = [1.5, 0]
    ball[7] = target
    args = dict(target_xyz=target, kick_origin_xy=origin, earliest_return_substep=0)
    assert strict_one_touch_return(ball, velocity, force, failed, **args)["strict_one_touch_return"]
    force[2, 1, 0] = 2
    assert not strict_one_touch_return(ball, velocity, force, failed, **args)[
        "strict_one_touch_return"
    ]
    force[2, 1, 0] = 0
    force[8, 1, 2] = 1.1
    assert not strict_one_touch_return(ball, velocity, force, failed, **args)[
        "strict_one_touch_return"
    ]
    force[8, 1, 2], force[2, 0, 0] = 0, 0
    assert not strict_one_touch_return(ball, velocity, force, failed, **args)[
        "strict_one_touch_return"
    ]
