import numpy as np
import pytest

from rosclaw_soccer.rsi.native_predictive_tracking import (
    NativePredictiveTracker,
    PredictiveTrackingBudget,
    actuator_controls,
    interpolate_residuals,
)


def test_three_knot_interpolation_and_nonidentity_actuator_mapping():
    knots = np.zeros((2, 3, 29))
    knots[:, 1] = 1
    knots[:, 2] = 2
    targets = interpolate_residuals(knots, 5)
    assert targets[0, :, 0].tolist() == [0, 0.5, 1, 1.5, 2]
    ids = tuple(reversed(range(29)))
    targets = np.arange(29, dtype=np.float64)[None, None, :] + np.zeros((2, 3, 29))
    control = actuator_controls(targets, ids)
    assert control.shape == (2, 30, 29)
    assert np.array_equal(control[0, 0], np.arange(28, -1, -1))
    assert np.array_equal(control[:, 0], control[:, 9])


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(samples=7),
        dict(samples=129),
        dict(samples=True),
        dict(horizon_controls=2),
        dict(horizon_controls=26),
        dict(improvements=4),
        dict(threads=0),
        dict(threads=17),
        dict(seed=-1),
        dict(sigma_rad=0),
        dict(max_delta_rad=0.6),
        dict(sigma_rad=np.nan),
    ],
)
def test_bounded_teacher_allocation(kwargs):
    with pytest.raises(ValueError):
        PredictiveTrackingBudget(**kwargs)


@pytest.mark.parametrize("task_space", [False, True])
def test_native_threaded_rollouts_do_not_mutate_evaluation_body(task_space):
    import mujoco

    # A native toy articulation only; this is not a humanoid or football pass.
    bodies = []
    for i in range(29):
        bodies.append(
            f'<body name="link{i}" pos="0 0 .02"><joint name="j{i}" '
            'type="hinge" range="-1 1" armature=".1"/><geom type="sphere" size=".01" '
            'mass=".01" contype="0" conaffinity="0"/>'
        )
    joints = "".join(bodies) + "</body>" * 29
    actuators = "".join(
        f'<position name="a{i}" joint="j{i}" kp="20" kv="1"/>' for i in reversed(range(29))
    )
    model = mujoco.MjModel.from_xml_string(
        '<mujoco><compiler angle="radian"/>'
        '<option timestep=".002" gravity="0 0 0"/><worldbody><body><freejoint name="root"/>'
        '<geom type="sphere" size=".1" mass="1" contype="0" conaffinity="0"/>'
        + joints
        + "</body></worldbody><actuator>"
        + actuators
        + "</actuator></mujoco>"
    )
    data = mujoco.MjData(model)
    data.ctrl[:] = 0.123
    mujoco.mj_forward(model, data)
    snapshot = [x.copy() for x in (data.qpos, data.qvel, data.ctrl, data.qacc_warmstart)]
    tracker = NativePredictiveTracker(
        model,
        joint_names_hg=tuple(f"j{i}" for i in range(29)),
        actuator_ids_hg=tuple(reversed(range(29))),
        root_joint_name="root",
        budget=PredictiveTrackingBudget(samples=8, horizon_controls=3, improvements=1, threads=1),
        tracked_body_names=("link28",) if task_space else (),
    )
    try:
        if task_space:
            with pytest.raises(ValueError):
                tracker.plan(
                    data, np.zeros((3, 29)), np.zeros((3, 3)), np.tile([1.0, 0, 0, 0], (3, 1))
                )
            assert tracker.requested_planning_substeps == 0
        result = tracker.plan(
            data,
            np.zeros((3, 29)),
            np.zeros((3, 3)),
            np.tile([1.0, 0, 0, 0], (3, 1)),
            reference_tracked_position=np.tile(data.xpos[model.body("link28").id], (3, 1, 1))
            if task_space
            else None,
        )
        assert np.isfinite(result["target_hg"]).all()
        assert result["best_cost"] <= result["nominal_cost"]
        assert tracker.requested_planning_substeps == 240
        assert tracker.observed_advancing_substeps == 240
        assert tracker.kinematic_evaluations == (24 if task_space else 0)
        assert result["new_optimizer_steps"] == result["new_neural_calls"] == 0
        assert all(
            np.array_equal(before, after)
            for before, after in zip(
                snapshot, (data.qpos, data.qvel, data.ctrl, data.qacc_warmstart), strict=True
            )
        )
    finally:
        tracker.close()
