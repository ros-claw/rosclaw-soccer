import mujoco
import numpy as np
import pytest

from rosclaw_soccer.rsi.contact_point_kinematics import contact_point_relative_velocity


def test_relative_point_velocity_uses_both_body_twists():
    model = mujoco.MjModel.from_xml_string("""<mujoco><worldbody>
      <body name="player" pos="0 0 1"><freejoint/><geom type="sphere" size="0.1" mass="1"/></body>
      <body name="ball" pos="0.2 0 1"><freejoint/><geom type="sphere" size="0.1" mass="1"/></body>
    </worldbody></mujoco>""")
    data = mujoco.MjData(model)
    data.qvel[:3] = [0.5, 0, 0]
    data.qvel[6:9] = [-1.5, 1, 0]
    mujoco.mj_forward(model, data)
    result = contact_point_relative_velocity(
        model,
        data,
        np.array([0.1, 0, 1.0]),
        np.array([1.0, 0, 0]),
        int(model.body("ball").id),
        int(model.body("player").id),
    )
    assert result["relative_velocity_world_mps"] == pytest.approx([-2.0, 1.0, 0.0])
    assert result["normal_speed_abs_mps"] == pytest.approx(2.0)
    assert result["tangent_speed_mps"] == pytest.approx(1.0)
    assert result["force_reconstructed"] is False
    with pytest.raises(ValueError, match="contact point"):
        contact_point_relative_velocity(
            model,
            data,
            np.zeros(3),
            np.zeros(3),
            int(model.body("ball").id),
            int(model.body("player").id),
        )
