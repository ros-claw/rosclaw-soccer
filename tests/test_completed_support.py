import mujoco
import numpy as np
import pytest

from rosclaw_soccer.sim.completed_support import CompletedGroundSupport


def fixture():
    model = mujoco.MjModel.from_xml_string("""<mujoco><option timestep=".002"/>
    <worldbody><geom name="floor" type="plane" size="2 2 .1"/>
    <body name="pelvis" pos="0 0 .1"><freejoint/>
    <geom name="left" type="sphere" size=".1" pos="0 .15 0" mass="1"/>
    <geom name="right" type="sphere" size=".1" pos="0 -.15 0" mass="1"/>
    </body><body pos="1 0 .1"><freejoint/><geom name="ball" type="sphere" size=".1" mass=".43"/>
    </body></worldbody></mujoco>""")
    data = mujoco.MjData(model)
    reader = CompletedGroundSupport(
        model,
        pelvis_body=1,
        left_foot_geoms=frozenset({1}),
        right_foot_geoms=frozenset({2}),
        ground_geoms=frozenset({0}),
    )
    return model, data, reader


def test_completed_forces_do_not_refresh_or_mutate_solver(monkeypatch):
    model, data, reader = fixture()
    for _ in range(200):
        mujoco.mj_step(model, data)
    fields = ("qpos", "qvel", "qacc_warmstart", "qfrc_constraint", "efc_force", "geom_xpos")
    before = {k: getattr(data, k).copy() for k in fields}

    def forbidden(*args):
        raise AssertionError("support reader refreshed physics")

    monkeypatch.setattr(mujoco, "mj_forward", forbidden)
    forces = reader.read(data)
    assert np.all(forces > 0)
    assert forces.sum() == pytest.approx(2 * 9.81, rel=0.02)
    assert all(np.array_equal(value, getattr(data, key)) for key, value in before.items())
    forces[:] = 100
    assert reader.read(data).sum() == pytest.approx(2 * 9.81, rel=0.02)


def test_foreign_model_and_dynamic_ground_rejected():
    model, data, reader = fixture()
    _, foreign, _ = fixture()
    with pytest.raises(ValueError, match="bound model"):
        reader.read(foreign)
    with pytest.raises(ValueError, match="static-ground"):
        CompletedGroundSupport(
            model,
            pelvis_body=1,
            left_foot_geoms=frozenset({1}),
            right_foot_geoms=frozenset({2}),
            ground_geoms=frozenset({3}),
        )
    with pytest.raises(ValueError, match="subtree"):
        CompletedGroundSupport(
            model,
            pelvis_body=1,
            left_foot_geoms=frozenset({3}),
            right_foot_geoms=frozenset({2}),
            ground_geoms=frozenset({0}),
        )


def test_nonfinite_force_is_rejected(monkeypatch):
    model, data, reader = fixture()
    mujoco.mj_step(model, data)

    def bad_force(model, data, index, wrench):
        wrench[:] = np.nan

    monkeypatch.setattr(mujoco, "mj_contactForce", bad_force)
    with pytest.raises(ValueError, match="nonfinite"):
        reader.read(data)
