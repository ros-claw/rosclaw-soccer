"""Tiny Euler contact checks, not G1 policy or task qualification."""

import mujoco
import numpy as np
import pytest

from rosclaw_soccer.physics.ball_contact_replay_diagnostic import ball_contact_samples


@pytest.mark.parametrize("ground", ["plane", "box"])
def test_force_on_ball_matches_generalized_force_and_preintegration_velocity(ground):
    geometry = (
        '<geom type="plane" size="1 1 .1"/>'
        if ground == "plane"
        else '<geom type="box" size="1 1 .05" pos="0 0 -.05"/>'
    )
    model = mujoco.MjModel.from_xml_string(
        '<mujoco><option timestep=".002" integrator="Euler" solver="Newton" iterations="100"/>'
        "<worldbody>" + geometry + '<body pos="0 0 .095"><freejoint/>'
        '<geom name="ball" type="sphere" size=".1" mass=".43"/></body></worldbody></mujoco>'
    )
    data = mujoco.MjData(model)
    data.qvel[2] = -0.3
    mujoco.mj_forward(model, data)
    before = data.qvel.copy()
    mujoco.mj_step(model, data)
    saved = (data.qpos.copy(), data.qvel.copy(), data.time)
    rows = ball_contact_samples(model, data, int(model.geom("ball").id), before)
    assert rows
    force = np.sum([r["force_on_ball_world_n"] for r in rows], axis=0)
    np.testing.assert_allclose(force, data.qfrc_constraint[:3], atol=1e-9, rtol=0)
    assert force[2] > 0
    assert rows[0]["signed_normal_relative_velocity_m_s"] == pytest.approx(-0.3)
    assert abs(data.qvel[2] - before[2]) > 0.01  # Post-integration velocity is a different clock.
    np.testing.assert_array_equal(data.qpos, saved[0])
    np.testing.assert_array_equal(data.qvel, saved[1])
    assert data.time == saved[2]
    if ground == "box":
        assert rows[0]["geom1"] == model.geom("ball").id
    else:
        assert rows[0]["geom2"] == model.geom("ball").id


def test_wrong_integrator_or_velocity_is_rejected():
    model = mujoco.MjModel.from_xml_string(
        '<mujoco><option integrator="RK4"/><worldbody><body><freejoint/>'
        '<geom name="ball" type="sphere" size=".1"/></body></worldbody></mujoco>'
    )
    data = mujoco.MjData(model)
    with pytest.raises(ValueError):
        ball_contact_samples(model, data, int(model.geom("ball").id), data.qvel.copy())
