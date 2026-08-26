from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from rosclaw_soccer.world.field import (
    G1TrainingGoalSpec,
    build_g1_four_player_two_ball_stadium_model,
)


@pytest.mark.integration
def test_four_player_stadium_has_two_independent_physical_footballs() -> None:
    mujoco = pytest.importorskip("mujoco")
    asset_root = Path("/code/rosclaw/phase4_references/RoboNaldo/RoboNaldo_Deploy")
    if not asset_root.is_dir():
        pytest.skip("external qualified RoboNaldo assets are unavailable")
    goal = G1TrainingGoalSpec(
        plane_x_m=7.5,
        width_m=7.32,
        height_m=2.44,
        regulation_field_enabled=True,
    )
    second_ball_origin = (1.25, -2.40, goal.ball_radius_m)
    model = build_g1_four_player_two_ball_stadium_model(
        asset_root,
        passer_origin_m=(5.10, -0.164, 0.0),
        goalkeeper_origin_m=(7.02, 0.0, 0.0),
        second_striker_origin_m=(0.0, -2.40, 0.0),
        first_ball_origin_m=(3.895, -2.84, goal.ball_radius_m),
        second_ball_origin_m=second_ball_origin,
        spec=goal,
    )
    data = mujoco.MjData(model)
    second_joint = int(model.joint("second_ball_free").id)
    second_qpos = int(model.jnt_qposadr[second_joint])
    second_dof = int(model.jnt_dofadr[second_joint])

    assert model.nu == 116
    assert model.body("second_striker_pelvis").id >= 0
    assert model.body("second_ball").id >= 0
    assert model.geom("second_ball_geom").id >= 0
    np.testing.assert_allclose(data.qpos[second_qpos : second_qpos + 3], second_ball_origin)
    np.testing.assert_allclose(model.body_ipos[int(model.body("second_ball").id)], 0.0)
    np.testing.assert_allclose(
        model.geom_size[int(model.geom("second_ball_geom").id), 0],
        goal.ball_radius_m,
    )
    np.testing.assert_allclose(model.body_mass[int(model.body("second_ball").id)], 0.41)
    np.testing.assert_allclose(model.dof_damping[second_dof : second_dof + 3], 0.02)
    np.testing.assert_allclose(model.dof_damping[second_dof + 3 : second_dof + 6], 0.00002)
    first_pair = int(model.pair("ball_floor").id)
    second_pair = int(model.pair("second_ball_floor").id)
    np.testing.assert_allclose(model.pair_friction[second_pair], model.pair_friction[first_pair])
    np.testing.assert_allclose(model.pair_solref[second_pair], model.pair_solref[first_pair])
    np.testing.assert_allclose(model.pair_solimp[second_pair], model.pair_solimp[first_pair])


def test_second_ball_origin_must_be_on_the_declared_pitch() -> None:
    asset_root = Path("/code/rosclaw/phase4_references/RoboNaldo/RoboNaldo_Deploy")
    if not asset_root.is_dir():
        pytest.skip("external qualified RoboNaldo assets are unavailable")
    with pytest.raises(ValueError, match="second football origin"):
        build_g1_four_player_two_ball_stadium_model(
            asset_root,
            passer_origin_m=(5.10, -0.164, 0.0),
            goalkeeper_origin_m=(7.02, 0.0, 0.0),
            second_striker_origin_m=(0.0, -2.40, 0.0),
            first_ball_origin_m=(3.895, -2.84, 0.115),
            second_ball_origin_m=(1.25, -40.0, 0.115),
            spec=G1TrainingGoalSpec(regulation_field_enabled=True),
        )
