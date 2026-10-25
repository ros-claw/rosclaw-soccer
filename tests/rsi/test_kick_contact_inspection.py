import numpy as np
import pytest

from rosclaw_soccer.rsi.kick_contact_inspection import inspect_ball_geometry


def fixture():
    import mujoco

    model = mujoco.MjModel.from_xml_string("""
    <mujoco><option gravity="0 0 0"/><worldbody>
      <body name="soccer_ball" pos="0 0 1.15"><freejoint/>
        <geom name="ball" type="sphere" size=".11" mass=".43"/></body>
      <body name="a/pelvis" pos="0 0 1"><freejoint/>
        <geom name="pelvis" type="sphere" size=".1" mass="1"/>
        <body name="a/left_ankle_roll_link" pos=".5 0 0">
          <geom type="sphere" size=".01" contype="0" conaffinity="0"/></body>
        <body name="a/right_ankle_roll_link" pos="-.5 0 0">
          <geom type="sphere" size=".01" contype="0" conaffinity="0"/></body>
      </body></worldbody></mujoco>
    """)
    qpos = np.repeat(mujoco.MjData(model).qpos[None], 3, axis=0)
    qpos[1:, 0] = 10  # Post-integration state no longer overlaps the pelvis.
    return model, qpos


def test_inspects_preintegration_clock_and_never_reconstructs_solved_force():
    model, qpos = fixture()
    before = qpos.copy()
    report = inspect_ball_geometry(model, qpos, native_index=1)
    assert report["inspected_preintegration_qpos_index"] == 0
    assert any(
        x["body"] == "a/pelvis" and not x["is_allowed_foot"] for x in report["geometric_contacts"]
    )
    assert not inspect_ball_geometry(model, qpos, native_index=2)["geometric_contacts"]
    assert not report["solved_contact_forces_reconstructed"]
    assert not report["model_clock_advanced"]
    np.testing.assert_array_equal(qpos, before)


@pytest.mark.parametrize("index", [0, -1, 3, True, 1.0])
def test_requires_available_preceding_measured_state(index):
    model, qpos = fixture()
    with pytest.raises(ValueError):
        inspect_ball_geometry(model, qpos, native_index=index)
