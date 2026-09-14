import dataclasses

import mujoco
import numpy as np
import pytest

from rosclaw_soccer.physics.contact_wrench import read_geom_contact_wrenches


def world(tilt=0.0):
    model = mujoco.MjModel.from_xml_string(
        '<mujoco><compiler angle="radian"/><option cone="elliptic"/>'
        '<worldbody><geom name="floor" type="plane" size="2 2 .1" '
        f'euler="0 {tilt} 0"/>'
        '<body pos="0 0 .09"><freejoint/><geom name="ball" size=".1" mass=".43"/>'
        '</body><geom name="unused" pos="3 0 2" size=".1"/></worldbody></mujoco>'
    )
    data = mujoco.MjData(model)
    data.qvel[:2] = [1.0, 0.4]
    mujoco.mj_forward(model, data)
    return model, data


@pytest.mark.parametrize("tilt", [0.0, 0.3, -0.3])
def test_signed_world_force_matches_generalized_free_body_force(tilt):
    model, data = world(tilt)
    ball = read_geom_contact_wrenches(model, data, geom_id=int(model.geom("ball").id))
    floor = read_geom_contact_wrenches(model, data, geom_id=int(model.geom("floor").id))
    assert len(ball) == len(floor) == 1
    np.testing.assert_allclose(ball[0].force_world_n, data.qfrc_constraint[:3], atol=1e-10)
    np.testing.assert_allclose(ball[0].force_world_n, -np.array(floor[0].force_world_n))
    np.testing.assert_allclose(
        ball[0].normal_toward_geom_world, -np.array(floor[0].normal_toward_geom_world)
    )
    assert ball[0].contact_normal_force_n > 0
    assert ball[0].point_world_m == floor[0].point_world_m
    assert ball[0].signed_distance_m < 0


def test_read_is_detached_immutable_and_does_not_refresh_state():
    model, data = world()
    names = ("qpos", "qvel", "qacc", "qacc_warmstart", "qfrc_constraint", "ctrl")
    before = {name: getattr(data, name).copy() for name in names}
    result = read_geom_contact_wrenches(model, data, geom_id=int(model.geom("ball").id))
    for name in names:
        np.testing.assert_array_equal(getattr(data, name), before[name])
    with pytest.raises(dataclasses.FrozenInstanceError):
        result[0].geom_id = 0
    point = result[0].point_world_m
    data.contact[0].pos[:] = 0
    assert result[0].point_world_m == point


def test_no_contact_returns_empty():
    model, data = world()
    assert read_geom_contact_wrenches(model, data, geom_id=int(model.geom("unused").id)) == ()


@pytest.mark.parametrize("geom_id", [-1, 1000, True, 1.0, None])
def test_bad_geometry_id_refused(geom_id):
    model, data = world()
    with pytest.raises(ValueError, match="geometry ID"):
        read_geom_contact_wrenches(model, data, geom_id=geom_id)


def test_equal_shape_other_model_data_refused():
    model, _ = world()
    _, data = world()
    with pytest.raises(ValueError, match="exact supplied model"):
        read_geom_contact_wrenches(model, data, geom_id=1)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1.0, 3601.0])
def test_bad_clock_refused(value):
    model, data = world()
    data.time = value
    with pytest.raises(ValueError, match="simulation time"):
        read_geom_contact_wrenches(model, data, geom_id=1)


@pytest.mark.parametrize("field", ["frame", "pos", "dist"])
def test_nonfinite_contact_refused(field):
    model, data = world()
    if field == "dist":
        data.contact[0].dist = float("nan")
    else:
        getattr(data.contact[0], field)[0] = float("nan")
    with pytest.raises(ValueError, match="nonfinite"):
        read_geom_contact_wrenches(model, data, geom_id=int(model.geom("ball").id))


def test_nonorthogonal_frame_refused():
    model, data = world()
    data.contact[0].frame[:] = 0
    with pytest.raises(ValueError, match="contact frame"):
        read_geom_contact_wrenches(model, data, geom_id=int(model.geom("ball").id))


def test_reflected_frame_refused():
    model, data = world()
    data.contact[0].frame[0:3] *= -1
    with pytest.raises(ValueError, match="contact frame"):
        read_geom_contact_wrenches(model, data, geom_id=int(model.geom("ball").id))


def test_nonfinite_force_refused(monkeypatch):
    model, data = world()

    def corrupt_force(model, data, index, result):
        result[:] = float("nan")

    monkeypatch.setattr(mujoco, "mj_contactForce", corrupt_force)
    with pytest.raises(ValueError, match="contact wrench"):
        read_geom_contact_wrenches(model, data, geom_id=int(model.geom("ball").id))


def test_zero_force_contact_not_silently_dropped():
    model, data = world()
    data.efc_force[:] = 0
    result = read_geom_contact_wrenches(model, data, geom_id=int(model.geom("ball").id))
    assert len(result) == 1
    assert result[0].force_world_n == (0.0, 0.0, 0.0)


def test_flex_or_invalid_peer_contact_refused():
    model, data = world()
    ball_id = int(model.geom("ball").id)
    other_index = 0 if int(data.contact[0].geom[1]) == ball_id else 1
    data.contact[0].geom[other_index] = -1
    with pytest.raises(ValueError, match="rigid geometries"):
        read_geom_contact_wrenches(model, data, geom_id=ball_id)
