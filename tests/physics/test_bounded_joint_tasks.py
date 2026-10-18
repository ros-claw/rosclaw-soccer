import numpy as np
import pytest

from rosclaw_soccer.physics.bounded_joint_tasks import LinearJointTask, solve_bounded_joint_tasks


def proposal(*, tasks=None, current=None, nominal=None, previous=None, limits=None):
    jac = np.zeros((3, 29))
    jac[0, 0] = 1
    return solve_bounded_joint_tasks(
        np.zeros(29) if current is None else current,
        np.zeros(29) if nominal is None else nominal,
        np.zeros(29) if previous is None else previous,
        np.tile([-2.0, 2.0], (29, 1)) if limits is None else limits,
        (LinearJointTask("swing", jac, np.array([0.1, 0.0, 0.0]), 1.0),)
        if tasks is None
        else tasks,
        observation_step_cap_rad=0.08,
        command_step_cap_rad=0.04,
        nominal_trust_rad=0.35,
        posture_cost=0.01,
    )


def test_motion_and_command_bounds_are_not_posthoc_physics_qualification():
    p = proposal()
    assert p.target_rad is not None
    assert p.target_rad[0] == pytest.approx(0.04)
    assert np.max(abs(np.asarray(p.target_rad)[1:])) == 0
    assert p.physically_qualified is False
    assert p.task_residual_m["swing"][0] == pytest.approx(-0.06)


def test_joint_limits_and_nominal_trust_intersect_command_limits():
    limits = np.tile([-2.0, 2.0], (29, 1))
    limits[0, 1] = 0.02
    p = proposal(limits=limits)
    assert p.target_rad is not None and p.target_rad[0] == pytest.approx(0.02)


def test_conflicting_objectives_share_one_joint_solution():
    jac = np.zeros((3, 29))
    jac[0, 0] = 1
    tasks = (
        LinearJointTask("swing", jac, np.array([0.1, 0.0, 0.0]), 1.0),
        LinearJointTask("stance", jac, np.zeros(3), 100.0),
    )
    p = proposal(tasks=tasks)
    assert p.target_rad is not None and 0 < p.target_rad[0] < 0.002
    assert p.task_residual_m["swing"][0] < -0.09
    assert p.physically_qualified is False


def test_infeasible_bounds_abstain_instead_of_expanding_or_clipping_intent():
    p = proposal(current=np.ones(29))
    assert p.target_rad is None and p.weighted_cost is None
    assert p.status == "ABSTAIN_INFEASIBLE_BOUND_INTERSECTION"


def test_fixed_bound_variables_do_not_require_invalid_strict_solver_bounds():
    limits = np.tile([-2.0, 2.0], (29, 1))
    limits[0] = [-0.2, 0.1]
    previous = np.zeros(29)
    previous[0] = 0.14
    p = proposal(limits=limits, previous=previous, current=np.full(29, 0.03))
    assert p.target_rad is not None and p.target_rad[0] == pytest.approx(0.1)


def test_nominal_is_a_pose_objective_not_the_jacobian_origin():
    jac = np.zeros((3, 29))
    jac[0, 0] = 1
    zero = np.zeros(29)
    q = zero.copy()
    q[0] = 0.1
    p = proposal(
        tasks=(LinearJointTask("foot", jac, np.zeros(3), 100.0),),
        current=q,
        nominal=q + 0.01,
        previous=q,
    )
    assert p.target_rad is not None and abs(p.target_rad[0] - 0.1) < 1e-4


@pytest.mark.parametrize("kind", ["nan", "dtype", "shape", "limits"])
def test_bad_body_arrays_rejected(kind):
    q = np.zeros(29)
    limits = np.tile([-2.0, 2.0], (29, 1))
    if kind == "nan":
        q[0] = np.nan
    elif kind == "dtype":
        q = q.astype(np.float32)
    elif kind == "shape":
        q = q[:28]
    else:
        limits[0] = [1, 0]
    with pytest.raises(ValueError):
        proposal(current=q, limits=limits)


def test_untyped_task_objects_rejected_before_reading_their_fields():
    with pytest.raises(ValueError):
        proposal(tasks=(object(),))


def test_invalid_task_arrays_rejected_without_attribute_errors():
    with pytest.raises(ValueError):
        proposal(tasks=(LinearJointTask("bad", None, np.zeros(3), 1.0),))
