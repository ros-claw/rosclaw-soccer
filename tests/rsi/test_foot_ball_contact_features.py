import numpy as np
import pytest

from rosclaw_soccer.rsi.foot_ball_contact_features import measured_foot_ball_contact_features


def world():
    mujoco = pytest.importorskip("mujoco")
    spheres = "".join(
        f'<geom type="sphere" size="0.005" pos="{x} {y} 0"/>'
        for x, y in ((0.05, 0.02), (0.05, -0.02), (-0.05, 0.02), (-0.05, -0.02))
    )
    model = mujoco.MjModel.from_xml_string(
        "<mujoco><worldbody>"
        f'<body name="left_ankle_roll_link" pos="0 0.1 0.03"><freejoint/>{spheres}</body>'
        f'<body name="right_ankle_roll_link" pos="0 -0.1 0.03"><freejoint/>{spheres}</body>'
        '<body name="ball" pos="1 0 0.11"><freejoint/>'
        '<geom name="ball_geom" type="sphere" size="0.11" mass="0.43"/></body>'
        "</worldbody></mujoco>"
    )
    return mujoco, model, mujoco.MjData(model)


def test_current_point_geometry_velocity_translation_and_nonmutation():
    mujoco, model, data = world()
    data.qvel[0] = 0.3
    data.qvel[12] = 1.2
    mujoco.mj_forward(model, data)
    initial = (
        data.qpos.copy(),
        data.qvel.copy(),
        data.qacc_warmstart.copy(),
        model.geom_size.copy(),
        data.time,
    )
    features = measured_foot_ball_contact_features(model, data)
    assert features.shape == (56,) and not features.flags.writeable
    points = features.reshape(8, 7)
    np.testing.assert_allclose(points[0, :3], [0.95, -0.12, 0.08], atol=1e-7)
    np.testing.assert_allclose(points[:4, 3], 0.9, atol=1e-7)
    np.testing.assert_allclose(points[4:, 3], 1.2, atol=1e-7)
    np.testing.assert_allclose(
        points[:, 6], np.linalg.norm(points[:, :3], axis=1) - 0.115, atol=1e-7
    )
    for actual, expected in zip(
        (data.qpos, data.qvel, data.qacc_warmstart, model.geom_size, data.time),
        initial,
        strict=True,
    ):
        np.testing.assert_array_equal(actual, expected)
    for address in (0, 7, 14):
        data.qpos[address : address + 3] += [4, -2, 0.5]
    mujoco.mj_forward(model, data)
    np.testing.assert_allclose(
        measured_foot_ball_contact_features(model, data), features, atol=1e-6
    )


def test_missing_sphere_collision_contract_is_rejected():
    mujoco, model, data = world()
    model.geom_contype[0] = model.geom_conaffinity[0] = 0
    mujoco.mj_forward(model, data)
    with pytest.raises(ValueError):
        measured_foot_ball_contact_features(model, data)
