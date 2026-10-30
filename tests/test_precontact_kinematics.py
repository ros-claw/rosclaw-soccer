import mujoco
import numpy as np
import pytest

from rosclaw_soccer.sim.precontact_kinematics import (
    PrecontactKinematics,
    cushion_shaping,
    observe_separated_geometry,
)


def test_separated_geometric_rate_and_no_state_mutation():
    model = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><body name="ball"><freejoint/>'
        '<geom name="ball" type="sphere" size=".1" mass="1"/></body>'
        '<body name="foot" pos="1 0 0"><freejoint/>'
        '<geom name="foot" type="sphere" size=".1" mass="1"/></body></worldbody></mujoco>'
    )
    data = mujoco.MjData(model)
    data.qvel[0] = 1.0
    q, v = data.qpos.copy(), data.qvel.copy()
    ball, foot = model.geom("ball").id, model.geom("foot").id
    observation = observe_separated_geometry(model, q, v, object_geom=ball, target_geoms=(foot,))
    assert observation.status == "SEPARATED"
    assert observation.distance_m == pytest.approx(0.8)
    assert observation.radial_separation_rate_mps == pytest.approx(-1.0)
    assert observation.nearest_geom == foot
    np.testing.assert_array_equal(q, data.qpos)
    np.testing.assert_array_equal(v, data.qvel)
    assert observation.hardware_authorized is False


def test_censored_states_and_shaping_remain_distinct():
    model = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><body name="a"><freejoint/>'
        '<geom name="a" type="sphere" size=".1" mass="1"/></body>'
        '<body name="b" pos="1 0 0"><freejoint/>'
        '<geom name="b" type="sphere" size=".1" mass="1"/></body></worldbody></mujoco>'
    )
    data = mujoco.MjData(model)
    a, b = model.geom("a").id, model.geom("b").id
    far = observe_separated_geometry(
        model,
        data.qpos.copy(),
        data.qvel.copy(),
        object_geom=a,
        target_geoms=(b,),
        distance_cap_m=0.5,
    )
    assert far.status == "DISTANCE_CENSORED" and cushion_shaping(far) is None
    data.qpos[7:10] = (0.1, 0, 0)
    contact = observe_separated_geometry(
        model, data.qpos.copy(), data.qvel.copy(), object_geom=a, target_geoms=(b,)
    )
    assert contact.status == "CONTACT_OR_OVERLAP" and cushion_shaping(contact) is None
    slow = PrecontactKinematics("SEPARATED", 0.1, -0.5, b)
    fast = PrecontactKinematics("SEPARATED", 0.1, -2.0, b)
    assert 0 < cushion_shaping(fast) < cushion_shaping(slow) < 1
    with pytest.raises(ValueError):
        observe_separated_geometry(
            model, data.qpos.copy(), data.qvel.copy(), object_geom=a, target_geoms=(b, b)
        )
