import numpy as np

from rosclaw_soccer.physics.bounded_joint_tasks import LinearJointTask
from rosclaw_soccer.rsi.foundation_delta_tasks import solve_foundation_delta


def test_increment_is_not_misrepresented_as_current_pose_bound():
    q = np.zeros(29)
    nominal = np.full(29, 0.5)
    nominal[28] = 3.0  # Not active; frozen original target must stay untouched.
    jac = np.zeros((3, 29))
    jac[0, 0] = 1
    target, report = solve_foundation_delta(
        q,
        nominal,
        q,
        np.tile([-2.0, 2.0], (29, 1)),
        (LinearJointTask("foot", jac, np.array([0.1, 0.0, 0.0]), 100.0),),
    )
    assert target[0] == 0.53
    np.testing.assert_array_equal(target[12:], nominal[12:])
    assert report["nominal_tracking_error_rad"] == 0.5
    assert report["measured_pose_ik"] is False
    assert not report["physical_response_proven"]


def test_increment_physical_limit_cannot_be_bought_with_large_parent_error():
    q = np.zeros(29)
    nominal = q.copy()
    nominal[0] = 3.0
    jac = np.zeros((3, 29))
    target, report = solve_foundation_delta(
        q,
        nominal,
        q,
        np.tile([-2.0, 2.0], (29, 1)),
        (LinearJointTask("foot", jac, np.zeros(3), 100.0),),
    )
    assert target is None and report["status"] == "ABSTAIN_INCREMENT_BOUND_INTERSECTION"


def test_opposed_foot_shin_tasks_share_increment_and_do_not_move_frozen_joints():
    q = np.zeros(29)
    jac = np.zeros((3, 29))
    jac[0, 0] = 1
    target, _ = solve_foundation_delta(
        q,
        q,
        q,
        np.tile([-2.0, 2.0], (29, 1)),
        (
            LinearJointTask("foot", jac, np.array([0.02, 0.0, 0.0]), 100.0),
            LinearJointTask("shin", jac, np.array([-0.02, 0.0, 0.0]), 100.0),
        ),
    )
    np.testing.assert_allclose(target, q, atol=1e-12)
