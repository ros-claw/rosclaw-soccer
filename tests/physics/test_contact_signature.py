from dataclasses import replace

import mujoco
import numpy as np
import pytest

from rosclaw_soccer.physics.contact_signature import (
    ContactSignature,
    compare_contacts,
    observe_contact,
)


def scene(pair: str = "", height: float = 0.1099):
    xml = f"""
    <mujoco>
      <worldbody>
        <geom name="floor" type="plane" size="1 1 0.01"/>
        <body name="ball" pos="0 0 {height}">
          <freejoint/>
          <geom name="ball_geom" type="sphere" size="0.11" mass="0.43"
                friction="0.8 0.01 0.0001"/>
        </body>
      </worldbody>
      {pair}
    </mujoco>
    """
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    ball = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "ball_geom")
    return model, data, floor, ball


def pair(dimension: int, rolling: float = 0.0001) -> str:
    return (
        '<contact><pair geom1="floor" geom2="ball_geom" '
        f'condim="{dimension}" friction="1 1 .01 {rolling} {rolling}" '
        'solref=".02 1" solimp=".9 .95 .001 .5 2"/></contact>'
    )


def test_named_and_automatic_3d_contacts_match_without_mutation():
    model, data, floor, ball = scene()
    before_q, before_v, before_time = data.qpos.copy(), data.qvel.copy(), data.time
    automatic = observe_contact(model, data, floor, ball)
    model2, data2, floor2, ball2 = scene(pair(3))
    named = observe_contact(model2, data2, ball2, floor2)

    assert automatic.dimension == 3
    assert len(automatic.active_friction) == 2
    assert compare_contacts(automatic, named).matches
    np.testing.assert_array_equal(data.qpos, before_q)
    np.testing.assert_array_equal(data.qvel, before_v)
    assert data.time == before_time


def test_six_dimensional_contact_and_clamped_zero_are_not_silently_matched():
    model, data, floor, ball = scene()
    original = observe_contact(model, data, floor, ball)
    model2, data2, floor2, ball2 = scene(pair(6, 0.0))
    rolling = observe_contact(model2, data2, floor2, ball2)

    assert rolling.dimension == 6
    assert rolling.active_friction[-2:] == pytest.approx((1e-5, 1e-5))
    result = compare_contacts(original, rolling)
    assert not result.matches
    assert "dimension" in result.mismatches
    assert result.activation_ceiling == "SIM_ONLY"
    changed = replace(rolling, active_friction=(0.5, *rolling.active_friction[1:]))
    assert compare_contacts(original, changed).mismatches == ("dimension", "shared_active_friction")


def test_missing_or_invalid_contact_fails_closed():
    model, data, floor, ball = scene(height=1.0)
    with pytest.raises(ValueError, match="exactly one"):
        observe_contact(model, data, floor, ball)
    with pytest.raises(ValueError, match="distinct"):
        observe_contact(model, data, floor, floor)
    with pytest.raises(ValueError, match="distinct"):
        observe_contact(model, data, -1, ball)

    # At exact tangency MuJoCo can report a geometric contact without an
    # active constraint; that is not evidence of an active contact model.
    tangent_model, tangent_data, tangent_floor, tangent_ball = scene(height=0.11)
    with pytest.raises(ValueError, match="exactly one"):
        observe_contact(tangent_model, tangent_data, tangent_floor, tangent_ball)


def test_multiple_contacts_are_not_collapsed_into_one_signature():
    model = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><geom name="floor" type="plane" size="1 1 .01"/>'
        '<body pos="0 0 .099"><freejoint/><geom name="box" type="box" '
        'size=".1 .1 .1" mass="1"/></body></worldbody></mujoco>'
    )
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    box = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "box")
    with pytest.raises(ValueError, match="exactly one"):
        observe_contact(model, data, floor, box)


def test_solver_and_active_friction_changes_are_reported():
    model, data, floor, ball = scene(pair(6))
    observed = observe_contact(model, data, floor, ball)
    modified = replace(
        observed, active_friction=(0.5, *observed.active_friction[1:]), solref=(0.03, 1.0)
    )
    result = compare_contacts(observed, modified)
    assert result.mismatches == ("active_friction", "solref")
    with pytest.raises(ValueError, match="tolerance"):
        compare_contacts(observed, modified, atol=float("nan"))
    with pytest.raises(ValueError, match="finite"):
        replace(observed, include_margin=float("nan"))
    with pytest.raises(ValueError, match="dimension"):
        ContactSignature(2, (1.0,), (0.02, 1.0), (0.0, 0.0), (0.9, 0.95, 0.001, 0.5, 2), 0.0)
