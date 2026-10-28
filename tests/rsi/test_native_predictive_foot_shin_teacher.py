import mujoco
import numpy as np
import pytest
from test_ball_surface_tasks import fixture

from rosclaw_soccer.rsi.native_predictive_foot_shin_teacher import NativePredictiveFootShinTeacher


def test_sim_teacher_uses_current_geometry_without_mutating_physics():
    model, data, layout = fixture()
    model.jnt_range[layout.joints] = [-2.0, 2.0]
    data.qpos[layout.bq : layout.bq + 3] = [0.1, 0.07, 0.2]
    data.qvel[layout.bv : layout.bv + 3] = [0.2, 0.1, 0.0]
    mujoco.mj_forward(model, data)
    before = [x.copy() for x in (data.qpos, data.qvel, data.ctrl, data.xfrc_applied)]
    teacher = NativePredictiveFootShinTeacher()
    result = teacher.propose(
        model=model, snapshot=data, layout=layout, nominal=np.zeros(29), enabled=True, frame=0
    )
    assert result["foot_index"] in (0, 1)
    assert not result["geometry_censored"]
    assert np.max(np.abs(result["applied_delta"])) <= 0.025 + 1e-12
    for actual, saved in zip(
        (data.qpos, data.qvel, data.ctrl, data.xfrc_applied), before, strict=True
    ):
        np.testing.assert_array_equal(actual, saved)
    assert data.time == 0
    with pytest.raises(ValueError):
        teacher.propose(
            model=model, snapshot=data, layout=layout, nominal=np.zeros(29), enabled=True, frame=0
        )


def test_far_ball_is_explicitly_censored_not_a_fake_surface():
    model, data, layout = fixture()
    model.jnt_range[layout.joints] = [-2.0, 2.0]
    data.qpos[layout.bq : layout.bq + 3] = [10, 0, 0]
    mujoco.mj_forward(model, data)
    result = NativePredictiveFootShinTeacher().propose(
        model=model,
        snapshot=data,
        layout=layout,
        nominal=np.zeros(29),
        enabled=True,
        frame=0,
    )
    assert result["geometry_censored"] and result["foot_index"] == -1
    assert not result["enabled"]
    np.testing.assert_array_equal(result["applied_delta"], np.zeros(29))
