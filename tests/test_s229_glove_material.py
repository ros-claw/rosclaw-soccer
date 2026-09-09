import numpy as np
import pytest

from rosclaw_soccer.world.goalkeeper_glove_material import GoalkeeperGloveMaterial


def model_with_gloves(*, omit_right_joint=False):
    import mujoco

    bodies = []
    for side in ("left", "right"):
        joint = (
            ""
            if side == "right" and omit_right_joint
            else f'<joint name="{side}_wrist_pitch_joint"/>'
        )
        bodies.append(
            f'<body name="{side}">{joint}<geom name="{side}_goalkeeper_glove" '
            'type="ellipsoid" size=".095 .05 .0325"/></body>'
        )
    return mujoco.MjModel.from_xml_string(
        "<mujoco><worldbody>" + "".join(bodies) + "</worldbody></mujoco>"
    )


def test_material_changes_response_but_not_reach_mass_or_positions():
    model = model_with_gloves()
    before = (model.geom_size.copy(), model.geom_pos.copy(), model.body_mass.copy())
    GoalkeeperGloveMaterial(0.0075, 0.15).apply(model, prefix="")
    np.testing.assert_allclose(model.geom_solref, [[0.0075, 0.15]] * 2)
    np.testing.assert_array_equal(model.geom_priority, [1, 1])
    np.testing.assert_allclose(model.jnt_margin, [0.08, 0.08])
    for a, b in zip(before, (model.geom_size, model.geom_pos, model.body_mass), strict=True):
        np.testing.assert_array_equal(a, b)


def test_material_resolves_all_targets_before_any_write():
    model = model_with_gloves(omit_right_joint=True)
    before = model.geom_solref.copy()
    with pytest.raises(KeyError):
        GoalkeeperGloveMaterial(0.0075, 0.15).apply(model, prefix="")
    np.testing.assert_array_equal(before, model.geom_solref)
    with pytest.raises(ValueError):
        GoalkeeperGloveMaterial(float("nan"), 0.15)
