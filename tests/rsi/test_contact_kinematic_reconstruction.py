import numpy as np
import pytest

from rosclaw_soccer.physics.contact_kinematic_reconstruction import (
    contact_kinematics,
    native_pre_step_state,
)


def model():
    import mujoco

    return mujoco.MjModel.from_xml_string(
        '<mujoco><option timestep=".002"/><worldbody>'
        '<body name="foot"><freejoint/><geom type="sphere" size=".03" mass="1"/></body>'
        '<body name="ball" pos=".139 0 0"><freejoint/>'
        '<geom type="sphere" size=".11" mass=".43"/></body></worldbody></mujoco>'
    )


def test_scratch_contact_point_jacobian_includes_rotation_and_never_steps(monkeypatch):
    import mujoco

    m = model()
    q, v = m.qpos0.copy(), np.zeros(m.nv)
    v[0], v[5], v[6] = 2.0, 3.0, 0.25
    before = q.copy(), v.copy()

    def no_step(*args):
        raise AssertionError("read-only reconstruction must not integrate")

    monkeypatch.setattr(mujoco, "mj_step", no_step)
    r = contact_kinematics(
        m,
        q,
        v,
        ball_body=int(m.body("ball").id),
        contact_bodies=(int(m.body("foot").id),),
        goal=np.array([4.0, 0.0, 0.0]),
    )
    assert len(r) == 1
    x = r[0]
    assert x["normal_towards_ball"] == pytest.approx([1, 0, 0])
    assert x["closing_normal_velocity_mps"] == pytest.approx(1.75)
    assert x["body_point_velocity_world"][1] == pytest.approx(3 * x["contact_point_world"][0])
    assert x["goal_heading_error_degrees"] == pytest.approx(0)
    assert (
        x["scratch_force_used"] is False and x["geometric_pair_is_recorded_force_source"] is False
    )
    assert np.array_equal(before[0], q) and np.array_equal(before[1], v)


def test_native_clock_shift_uses_previous_state_not_post_impact_label():
    q, v = np.array([1.0, 2.0]), np.array([3.0])
    qs, vs = np.array([[10.0, 20.0], [100.0, 200.0]]), np.array([[30.0], [300.0]])
    a, b = native_pre_step_state(q, v, qs, vs, 0)
    assert a.tolist() == [1, 2] and b.tolist() == [3]
    a, b = native_pre_step_state(q, v, qs, vs, 1)
    assert a.tolist() == [10, 20] and b.tolist() == [30]
    a.fill(99)
    assert qs[0].tolist() == [10, 20]


@pytest.mark.parametrize("index", [-1, 2, True, 1.0])
def test_invalid_native_index_rejected(index):
    with pytest.raises(ValueError):
        native_pre_step_state(np.zeros(2), np.zeros(1), np.zeros((2, 2)), np.zeros((2, 1)), index)


def test_no_contact_is_not_fabricated_and_nonfinite_or_wrong_dtype_rejected():
    m = model()
    q, v = m.qpos0.copy(), np.zeros(m.nv)
    q[7] = 2
    kwargs = dict(
        ball_body=int(m.body("ball").id),
        contact_bodies=(int(m.body("foot").id),),
        goal=np.array([4.0, 0.0, 0.0]),
    )
    assert contact_kinematics(m, q, v, **kwargs) == []
    with pytest.raises(ValueError):
        contact_kinematics(m, q.astype(np.float32), v, **kwargs)
    q[0] = np.nan
    with pytest.raises(ValueError):
        contact_kinematics(m, q, v, **kwargs)
