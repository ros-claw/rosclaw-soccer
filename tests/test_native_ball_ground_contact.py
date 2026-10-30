import mujoco
import numpy as np
import pytest

from rosclaw_soccer.physics.native_ball_ground_contact import inspect_ball_ground_contact


def model_with_pair(pair: str = ""):
    return mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><geom name="floor" type="plane" size="1 1 .1" condim="3"/>'
        '<body pos="0 0 .09"><freejoint/><geom name="ball" type="sphere" size=".1" '
        'mass="1" condim="3"/></body></worldbody>' + pair + "</mujoco>"
    )


def test_actual_three_dimensional_contact_is_not_rolling_constraint():
    model = model_with_pair()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    q, v = data.qpos.copy(), data.qvel.copy()
    result = inspect_ball_ground_contact(
        model, data, ball_geom=model.geom("ball").id, ground_geom=model.geom("floor").id
    )
    assert result.status == "NO_ROLLING_CONSTRAINT"
    assert result.effective_dimensions == (3,)
    assert result.effective_rolling_friction_m == (None,)
    np.testing.assert_array_equal(q, data.qpos)
    np.testing.assert_array_equal(v, data.qvel)


def test_explicit_pair_overrides_geom_dimensions():
    model = model_with_pair(
        '<contact><pair geom1="floor" geom2="ball" condim="6" '
        'friction="1 1 .005 .001 .002"/></contact>'
    )
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    result = inspect_ball_ground_contact(
        model, data, ball_geom=model.geom("ball").id, ground_geom=model.geom("floor").id
    )
    assert result.status == "ROLLING_CONSTRAINT_PRESENT"
    assert result.effective_dimensions == (6,)
    assert result.effective_rolling_friction_m[0] == pytest.approx((0.001, 0.002))
    assert result.hardware_authorized is False


def test_absent_contact_remains_unknown_and_bad_geoms_fail():
    model = model_with_pair()
    data = mujoco.MjData(model)
    data.qpos[2] = 1
    mujoco.mj_forward(model, data)
    result = inspect_ball_ground_contact(
        model, data, ball_geom=model.geom("ball").id, ground_geom=model.geom("floor").id
    )
    assert result.status == "NO_OBSERVED_CONTACT"
    assert result.effective_dimensions == ()
    with pytest.raises(ValueError):
        inspect_ball_ground_contact(model, data, ball_geom=0, ground_geom=0)


def test_inactive_near_contact_is_not_an_active_rolling_constraint():
    model = model_with_pair(
        '<contact><pair geom1="floor" geom2="ball" condim="6" '
        'margin=".2" gap=".1" friction="1 1 .005 .001 .002"/></contact>'
    )
    data = mujoco.MjData(model)
    data.qpos[2] = 0.25
    mujoco.mj_forward(model, data)
    result = inspect_ball_ground_contact(
        model, data, ball_geom=model.geom("ball").id, ground_geom=model.geom("floor").id
    )
    assert result.status == "INACTIVE_CONTACT_ONLY"
    assert result.effective_dimensions == ()
    assert result.inactive_contacts == 1
