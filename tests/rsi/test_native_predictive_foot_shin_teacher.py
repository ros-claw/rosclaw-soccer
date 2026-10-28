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


def test_early_candidate_responds_to_incoming_ball_without_editing_physics():
    model, data, layout = fixture()
    model.jnt_range[layout.joints] = [-2.0, 2.0]
    foot = model.geom("a/left_foot").id
    mujoco.mj_forward(model, data)
    data.qpos[layout.bq : layout.bq + 3] = data.geom_xpos[foot] + [1.2, 0.0, 0.0]
    data.qvel[layout.bv : layout.bv + 3] = [-3.0, 0.0, 0.0]
    mujoco.mj_forward(model, data)
    before = [x.copy() for x in (data.qpos, data.qvel, data.ctrl, data.xfrc_applied)]
    common = dict(
        model=model, snapshot=data, layout=layout, nominal=np.zeros(29), enabled=True, frame=0
    )
    old = NativePredictiveFootShinTeacher(axis_local_fallback=True).propose(**common)
    new = NativePredictiveFootShinTeacher(axis_local_fallback=True, early_intercept=True).propose(
        **common
    )
    lead = NativePredictiveFootShinTeacher(
        axis_local_fallback=True, early_intercept=True, contact_lead=True
    ).propose(**common)
    assert old["foot_index"] == -1
    assert new["early_active"] and new["foot_index"] in (0, 1)
    assert 0.03 <= new["early_eta_s"] <= 0.6
    assert new["foot_gap"] > 0.55
    assert lead["early_active"]
    assert not np.array_equal(lead["target"], new["target"])
    assert np.max(np.abs(new["applied_delta"])) <= 0.025 + 1e-12
    for actual, saved in zip(
        (data.qpos, data.qvel, data.ctrl, data.xfrc_applied), before, strict=True
    ):
        np.testing.assert_array_equal(actual, saved)


def test_explicit_reserve_is_bounded_and_does_not_mutate_model():
    model, data, layout = fixture()
    model.jnt_range[layout.joints] = [-2.0, 2.0]
    data.qpos[layout.bq : layout.bq + 3] = [0.1, 0.07, 0.2]
    mujoco.mj_forward(model, data)
    before = model.jnt_range.copy()
    teacher = NativePredictiveFootShinTeacher(joint_margin_rad=0.2)
    teacher.propose(
        model=model, snapshot=data, layout=layout, nominal=np.zeros(29), enabled=True, frame=0
    )
    np.testing.assert_array_equal(model.jnt_range, before)
    with pytest.raises(ValueError):
        NativePredictiveFootShinTeacher(joint_margin_rad=0.21)
