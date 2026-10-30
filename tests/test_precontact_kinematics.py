import mujoco
import numpy as np
import pytest

from rosclaw_soccer.sim.precontact_kinematics import (
    PrecontactKinematics,
    compare_separated_witnesses,
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


def test_pair_preserves_direction_and_no_arrival_for_receding_geometry():
    foot = PrecontactKinematics("SEPARATED", 0.1, -1.0, 1)
    body = PrecontactKinematics("SEPARATED", 0.14, -2.0, 2)
    result = compare_separated_witnesses(foot, body)
    assert result.status == "SEPARATED"
    assert result.gap_difference_m == pytest.approx(0.04)
    assert result.radial_difference_mps == pytest.approx(-1.0)
    assert result.linear_arrival_difference_s == pytest.approx(-0.03)
    assert result.hardware_authorized is False
    receding = PrecontactKinematics("SEPARATED", 0.14, 0.0, 2)
    assert compare_separated_witnesses(foot, receding).linear_arrival_difference_s is None


def test_pair_rejects_malformed_and_does_not_fabricate_censored_differences():
    foot = PrecontactKinematics("SEPARATED", 0.1, -1.0, 1)
    for other in (
        PrecontactKinematics("DISTANCE_CENSORED", 2.0, None, None),
        PrecontactKinematics("CONTACT_OR_OVERLAP", -0.01, None, 2),
    ):
        pair = compare_separated_witnesses(foot, other)
        assert pair.status == "UNOBSERVED_OR_CONTACT"
        assert pair.gap_difference_m is None and pair.radial_difference_mps is None
        assert pair.linear_arrival_difference_s is None
    with pytest.raises(ValueError):
        compare_separated_witnesses(foot, PrecontactKinematics("SEPARATED", 0.1, np.nan, 2))
    with pytest.raises(ValueError):
        compare_separated_witnesses(foot, PrecontactKinematics("MADE_UP", 0.1, 0, 2))
