import mujoco
import numpy as np
import pytest

from rosclaw_soccer.sim.physical_checkpoint import PhysicalCheckpoint
from rosclaw_soccer.sim.support_observation import observe_support


def fixture():
    model = mujoco.MjModel.from_xml_string("""
    <mujoco><worldbody><geom name="ground" type="plane" size="2 2 .1"/>
    <body name="pelvis" pos="0 0 .099"><freejoint/>
    <geom name="left" type="box" pos="0 .2 0" size=".1 .08 .1" mass="2"/>
    <geom name="right" type="box" pos="0 -.2 0" size=".1 .08 .1" mass="2"/>
    </body><body pos="1 0 1"><freejoint/><geom name="ball" type="sphere" size=".1"/></body>
    </worldbody></mujoco>""")
    data = mujoco.MjData(model)
    data.qvel[0] = 0.2
    bindings = dict(
        pelvis_body=1,
        left_foot_geoms=frozenset({1}),
        right_foot_geoms=frozenset({2}),
        ground_geoms=frozenset({0}),
    )
    return model, data, bindings


def test_double_support_without_live_state_mutation():
    model, data, bindings = fixture()
    before = PhysicalCheckpoint.capture(model, data)
    derived = data.xpos.copy()
    result = observe_support(model, data, **bindings)
    assert result.support_code == 3
    assert result.left_normal_force_n > 1 and result.right_normal_force_n > 1
    assert result.pelvis_velocity_mps[0] == pytest.approx(0.2)
    assert PhysicalCheckpoint.capture(model, data) == before
    np.testing.assert_array_equal(data.xpos, derived)


def test_airborne_is_not_support():
    model, data, bindings = fixture()
    data.qpos[2] = 1
    assert observe_support(model, data, **bindings).support_code == 0


@pytest.mark.parametrize("bad", [0, -1, float("nan"), float("inf"), True])
def test_invalid_threshold(bad):
    model, data, bindings = fixture()
    with pytest.raises(ValueError, match="threshold"):
        observe_support(model, data, **bindings, force_threshold_n=bad)


def test_ball_cannot_be_ground_or_foot():
    model, data, bindings = fixture()
    with pytest.raises(ValueError, match="static world"):
        observe_support(model, data, **(bindings | dict(ground_geoms=frozenset({3}))))
    with pytest.raises(ValueError, match="subtree"):
        observe_support(model, data, **(bindings | dict(left_foot_geoms=frozenset({3}))))


def test_overlap_rejected():
    model, data, bindings = fixture()
    with pytest.raises(ValueError, match="disjoint"):
        observe_support(model, data, **(bindings | dict(right_foot_geoms=frozenset({1}))))
