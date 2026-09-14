"""Other actors' ground contacts must not veto this robot's airborne evidence."""

import numpy as np
import pytest

from rosclaw_soccer.sim.support_forces import measure_support_forces

mujoco = pytest.importorskip("mujoco")


def fixture(height=1.0):
    model = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><geom name="floor" type="plane" size="10 10 .1"/>'
        f'<body name="robot" pos="0 0 {height}"><freejoint/>'
        '<geom size=".1" mass="1"/><body name="foot" pos="0 0 -.2">'
        '<geom size=".1" mass="1"/></body></body>'
        '<body name="ball" pos="2 0 .09"><freejoint/>'
        '<geom size=".1" mass=".43"/></body></worldbody></mujoco>'
    )
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    args = dict(
        robot_root_body_id=model.body("robot").id,
        support_body_ids=(model.body("foot").id,),
        ground_geom_ids=(model.geom("floor").id,),
    )
    return model, data, args


def test_ball_floor_contact_does_not_count_as_robot_support():
    model, data, args = fixture()
    assert data.ncon > 0
    before = (data.qpos.copy(), data.qvel.copy(), data.ctrl.copy(), data.time)
    result = measure_support_forces(model, data, **args)
    assert result.support_force_n == result.other_robot_ground_force_n == 0
    assert result.support_contacts == result.other_robot_ground_contacts == 0
    for original, current in zip(before[:3], (data.qpos, data.qvel, data.ctrl), strict=True):
        np.testing.assert_array_equal(original, current)
    assert data.time == before[3]


def test_supported_foot_is_counted_without_ball_force():
    model, data, args = fixture(0.29)
    result = measure_support_forces(model, data, **args)
    assert result.support_force_n > 0
    assert result.support_contacts > 0
    assert result.other_robot_ground_force_n == 0


def test_other_robot_body_ground_contact_is_counted():
    model, data, args = fixture(0.09)
    model.body_pos[model.body("foot").id, 2] = 0.2
    mujoco.mj_forward(model, data)
    result = measure_support_forces(model, data, **args)
    assert result.other_robot_ground_force_n > 0
    assert result.other_robot_ground_contacts > 0


@pytest.mark.parametrize(
    "field,value",
    [
        ("robot_root_body_id", 0),
        ("robot_root_body_id", True),
        ("support_body_ids", ()),
        ("support_body_ids", (-1,)),
        ("ground_geom_ids", ()),
        ("ground_geom_ids", (999,)),
    ],
)
def test_invalid_scope_rejects(field, value):
    model, data, args = fixture()
    args[field] = value
    with pytest.raises(ValueError):
        measure_support_forces(model, data, **args)


def test_foreign_support_and_own_ground_reject():
    model, data, args = fixture()
    with pytest.raises(ValueError, match="outside"):
        measure_support_forces(
            model, data, **{**args, "support_body_ids": (model.body("ball").id,)}
        )
    with pytest.raises(ValueError, match="belongs"):
        measure_support_forces(model, data, **{**args, "ground_geom_ids": (1,)})


@pytest.mark.parametrize("value", [np.nan, np.inf, 1e308])
def test_nonfinite_force_rejects(monkeypatch, value):
    model, data, args = fixture(0.29)

    def invalid_force(model, data, index, force):
        force[:] = value

    monkeypatch.setattr(mujoco, "mj_contactForce", invalid_force)
    with pytest.raises(ValueError, match="nonfinite"):
        measure_support_forces(model, data, **args)
