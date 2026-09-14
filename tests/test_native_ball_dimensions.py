import dataclasses
import math

import mujoco
import numpy as np
import pytest

from rosclaw_soccer.physics.native_ball_dimensions import inspect_native_ball_dimensions


def world(radius=0.115, mass=0.41, joint="<freejoint/>", geom_type="sphere"):
    model = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><body name="ball">'
        f'{joint}<geom name="ball_geom" type="{geom_type}" size="{radius} .1 .1" '
        f'mass="{mass}"/></body></worldbody></mujoco>'
    )
    return model, int(model.geom("ball_geom").id)


def test_legacy_ball_is_not_regulation_size_but_mass_is_allowed():
    model, geom = world()
    result = inspect_native_ball_dimensions(model, geom_id=geom)
    assert result.circumference_m == pytest.approx(0.7225663103256524)
    assert result.mass_in_ifab_range
    assert not result.circumference_in_ifab_range
    assert not result.size_and_mass_in_ifab_range


@pytest.mark.parametrize("circumference,mass", [(0.68, 0.41), (0.69, 0.43), (0.70, 0.45)])
def test_selected_values_and_exact_boundaries(circumference, mass):
    model, geom = world(circumference / (2 * math.pi), mass)
    assert inspect_native_ball_dimensions(model, geom_id=geom).size_and_mass_in_ifab_range


@pytest.mark.parametrize(
    "circumference,mass", [(0.679, 0.43), (0.701, 0.43), (0.69, 0.409), (0.69, 0.451)]
)
def test_out_of_range_is_reported_without_mutating_model(circumference, mass):
    model, geom = world(circumference / (2 * math.pi), mass)
    before = (model.geom_size.copy(), model.body_mass.copy(), model.body_inertia.copy())
    result = inspect_native_ball_dimensions(model, geom_id=geom)
    assert not result.size_and_mass_in_ifab_range
    for a, b in zip(before, (model.geom_size, model.body_mass, model.body_inertia), strict=True):
        np.testing.assert_array_equal(a, b)
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.radius_m = 0.1


@pytest.mark.parametrize("joint", ["", "<joint/>"])
def test_static_or_hinged_decoration_not_a_free_ball(joint):
    model, geom = world(joint=joint)
    with pytest.raises(ValueError, match="free joint"):
        inspect_native_ball_dimensions(model, geom_id=geom)


def test_box_not_called_spherical_ball():
    model, geom = world(geom_type="box")
    with pytest.raises(ValueError, match="spherical"):
        inspect_native_ball_dimensions(model, geom_id=geom)


@pytest.mark.parametrize("geom", [-1, 1000, True, 0.0, None])
def test_bad_geometry_id(geom):
    model, _ = world()
    with pytest.raises(ValueError, match="geometry ID"):
        inspect_native_ball_dimensions(model, geom_id=geom)


@pytest.mark.parametrize("field", ["geom_size", "body_mass", "body_inertia"])
def test_nonfinite_model_values_refused(field):
    model, geom = world()
    array = getattr(model, field)
    index = geom if field == "geom_size" else int(model.geom_bodyid[geom])
    array[index] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        inspect_native_ball_dimensions(model, geom_id=geom)
