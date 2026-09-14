import dataclasses

import mujoco
import numpy as np
import pytest

from rosclaw_soccer.physics.role_partition import partition_native_role


def world():
    return mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><geom name="floor" type="plane" size="2 2 .1"/>'
        '<body name="red"><freejoint/><geom name="red_torso" size=".1"/>'
        '<body name="red_foot" pos="0 0 .3"><joint name="red_joint"/>'
        '<geom name="misleading_blue_name" size=".05"/></body></body>'
        '<body name="blue" pos="1 0 0"><freejoint/><geom name="blue_torso" size=".1"/>'
        '<body name="blue_foot" pos="0 0 .3"><joint name="blue_joint"/>'
        '<geom name="misleading_red_name" size=".05"/></body></body>'
        '<body name="ball" pos="2 0 0"><freejoint/><geom size=".1"/></body>'
        '</worldbody><actuator><motor joint="blue_joint"/>'
        '<motor joint="red_joint"/></actuator></mujoco>'
    )


def test_ownership_uses_topology_not_geom_prefix_or_actuator_order():
    model = world()
    red = partition_native_role(model, root_body_name="red")
    blue = partition_native_role(model, root_body_name="blue")
    assert red.actuator_ids == (1,) and blue.actuator_ids == (0,)
    assert set(red.body_ids).isdisjoint(blue.body_ids)
    assert set(red.joint_ids).isdisjoint(blue.joint_ids)
    assert set(red.geom_ids).isdisjoint(blue.geom_ids)
    assert int(model.geom("misleading_blue_name").id) in red.geom_ids
    assert int(model.geom("floor").id) not in red.geom_ids + blue.geom_ids
    assert int(model.body("ball").id) not in red.body_ids + blue.body_ids


def test_child_partition_excludes_parent_joint_and_geom():
    model = world()
    child = partition_native_role(model, root_body_name="red_foot")
    assert len(child.body_ids) == len(child.geom_ids) == len(child.joint_ids) == 1
    assert child.actuator_ids == (1,)


def test_partition_does_not_mutate_model_and_ids_are_immutable():
    model = world()
    names = ("body_parentid", "geom_bodyid", "jnt_bodyid", "actuator_trnid", "geom_size")
    before = {name: getattr(model, name).copy() for name in names}
    result = partition_native_role(model, root_body_name="red")
    for name in names:
        np.testing.assert_array_equal(getattr(model, name), before[name])
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.root_body_id = 0


@pytest.mark.parametrize("name", ["", "missing", "world", None, "x" * 257])
def test_invalid_root_refused(name):
    with pytest.raises(ValueError):
        partition_native_role(world(), root_body_name=name)


def test_invalid_parent_order_refused():
    model = world()
    model.body_parentid[2] = 2
    with pytest.raises(ValueError, match="parent ordering"):
        partition_native_role(model, root_body_name="red")


def test_unsupported_transmission_refused_even_on_other_role():
    model = world()
    model.actuator_trntype[0] = int(mujoco.mjtTrn.mjTRN_SITE)
    with pytest.raises(ValueError, match="unsupported actuator"):
        partition_native_role(model, root_body_name="red")


def test_invalid_actuator_joint_binding_refused():
    model = world()
    model.actuator_trnid[0, 0] = model.njnt
    with pytest.raises(ValueError, match="joint binding"):
        partition_native_role(model, root_body_name="red")


def test_ball_partition_has_no_actuators():
    part = partition_native_role(world(), root_body_name="ball")
    assert len(part.body_ids) == len(part.joint_ids) == 1
    assert part.actuator_ids == ()


@pytest.mark.parametrize("field", ["geom_bodyid", "jnt_bodyid"])
def test_invalid_body_binding_cannot_be_silently_excluded(field):
    model = world()
    getattr(model, field)[-1] = model.nbody
    with pytest.raises(ValueError, match="body binding"):
        partition_native_role(model, root_body_name="red")
