import numpy as np
import pytest

from rosclaw_soccer.providers.g1.joint_contract import G1_DDS_JOINT_NAMES
from rosclaw_soccer.rsi.native_velocity_transfer import NativeVelocityLayout


@pytest.mark.parametrize("ball_first", [False, True])
def test_native_layout_keeps_actor_canonical_order_without_touching_the_body(ball_first):
    import mujoco

    prefix = "robot/" if ball_first else ""
    joints = (
        "".join(
            f'<body name="{prefix}link{i}"><joint name="{prefix}{name}" '
            'type="hinge" range="-2 2" armature=".1"/>'
            '<geom type="sphere" size=".01" mass=".01" contype="0" conaffinity="0"/>'
            for i, name in enumerate(G1_DDS_JOINT_NAMES)
        )
        + "</body>" * 29
    )
    robot = (
        f'<body name="{prefix}pelvis"><freejoint/><geom type="sphere" size=".1" mass="1"/>'
        + joints
        + "</body>"
    )
    ball_name = "ball/ball" if ball_first else "ball"
    ball = (
        f'<body name="{ball_name}"><freejoint/><geom type="sphere" size=".11" mass=".43"/></body>'
    )
    act = "".join(
        f'<position joint="{prefix}{name}" kp="20" kv="1"/>'
        for name in reversed(G1_DDS_JOINT_NAMES)
    )
    m = mujoco.MjModel.from_xml_string(
        '<mujoco><compiler angle="radian"/>'
        '<option gravity="0 0 0"/><worldbody>'
        + (ball + robot if ball_first else robot + ball)
        + "</worldbody><actuator>"
        + act
        + "</actuator></mujoco>"
    )
    layout = NativeVelocityLayout(m, robot_prefix=prefix, ball_body_name=ball_name)
    d = mujoco.MjData(m)
    d.qpos[layout.rq : layout.rq + 7] = [0.1, 0.2, 0.8, np.cos(0.3), 0, 0, np.sin(0.3)]
    d.qpos[layout.qi] = np.arange(29) * 0.01
    d.qpos[layout.bq : layout.bq + 3] = [3, 4, 0.11]
    d.qvel[layout.rv : layout.rv + 6] = [1, 2, 3, 0.2, -0.3, 0.4]
    d.qvel[layout.vi] = np.arange(29) * 0.02
    mujoco.mj_forward(m, d)
    before = d.qpos.copy(), d.qvel.copy()
    q, v = layout.canonical(d)
    assert q[36:39].tolist() == [3, 4, 0.11]
    np.testing.assert_array_equal(q[7:36], np.arange(29) * 0.01)
    np.testing.assert_array_equal(v[6:35], np.arange(29) * 0.02)
    np.testing.assert_allclose(layout.angular_velocity_body(d), [0.2, -0.3, 0.4], atol=1e-12)
    assert layout.ai == list(reversed(range(29)))
    assert layout.profile()["robot_mass_kg"] == pytest.approx(1.29)
    q[:] = 0
    v[:] = 0
    assert np.array_equal(d.qpos, before[0]) and np.array_equal(d.qvel, before[1])
