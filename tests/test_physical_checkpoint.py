from dataclasses import replace

import mujoco
import numpy as np
import pytest

from rosclaw_soccer.sim.contracts import hash_bytes
from rosclaw_soccer.sim.physical_checkpoint import PhysicalCheckpoint


def world():
    model = mujoco.MjModel.from_xml_string("""
    <mujoco><option timestep="0.002"/><size nuserdata="2"/>
    <worldbody><geom type="plane" size="2 2 .1"/>
    <body pos="0 0 .11"><freejoint/><geom type="sphere" size=".1" mass=".43"/>
    </body></worldbody></mujoco>""")
    data = mujoco.MjData(model)
    data.qvel[0] = 0.7
    data.userdata[:] = [3, 7]
    for _ in range(100):
        mujoco.mj_step(model, data)
    return model, data


def test_contact_checkpoint_replays_every_integration_value():
    model, data = world()
    snapshot = PhysicalCheckpoint.capture(model, data)
    restored = snapshot.restore(model)
    assert PhysicalCheckpoint.capture(model, restored) == snapshot
    # Actual contact physics, not an import-only smoke or final-position check.
    touched = False
    for _ in range(200):
        mujoco.mj_step(model, data)
        mujoco.mj_step(model, restored)
        touched |= data.ncon > 0
        assert PhysicalCheckpoint.capture(model, data) == PhysicalCheckpoint.capture(
            model, restored
        )
    assert touched


def test_changed_physics_rejected_even_with_same_dimensions():
    model, data = world()
    snapshot = PhysicalCheckpoint.capture(model, data)
    model.geom_friction[0, 0] *= 0.5
    with pytest.raises(ValueError, match="physics"):
        snapshot.restore(model)


@pytest.mark.parametrize("mutation", ["bytes", "version", "size", "nan"])
def test_bad_checkpoint_rejected(mutation):
    model, data = world()
    snapshot = PhysicalCheckpoint.capture(model, data)
    if mutation == "version":
        snapshot = replace(snapshot, mujoco_version="different")
    elif mutation == "bytes":
        snapshot = replace(snapshot, state_bytes=b"tampered")
    else:
        payload = (
            snapshot.state_bytes[:-8]
            if mutation == "size"
            else np.full(len(snapshot.state_bytes) // 8, np.nan, dtype="<f8").tobytes()
        )
        snapshot = replace(snapshot, state_bytes=payload, state_hash=hash_bytes(payload))
    with pytest.raises(ValueError):
        snapshot.restore(model)


def test_capture_rejects_nonfinite():
    model, data = world()
    data.qvel[0] = np.nan
    with pytest.raises(ValueError, match="nonfinite"):
        PhysicalCheckpoint.capture(model, data)


def test_wrong_model_data_pair_rejected_before_native_call():
    _, data = world()
    other, _ = world()
    with pytest.raises(ValueError, match="belong"):
        PhysicalCheckpoint.capture(other, data)


def test_control_activation_external_forces_and_mocap_are_restored():
    model = mujoco.MjModel.from_xml_string("""
    <mujoco><size nuserdata="2"/><worldbody>
    <body><joint name="hinge"/><geom type="capsule" size=".05 .2"/></body>
    <body mocap="true"><geom type="sphere" size=".02" contype="0" conaffinity="0"/></body>
    </worldbody><actuator><general joint="hinge" dyntype="filter" dynprm=".1"/></actuator>
    </mujoco>""")
    data = mujoco.MjData(model)
    data.ctrl[:] = 0.2
    data.act[:] = 0.3
    data.qacc_warmstart[:] = 0.4
    data.qfrc_applied[:] = 0.5
    data.xfrc_applied[1, :] = np.arange(6) * 0.01
    data.mocap_pos[0] = [0.3, 0.4, 0.5]
    data.userdata[:] = [0.6, 0.7]
    snapshot = PhysicalCheckpoint.capture(model, data)
    restored = snapshot.restore(model)
    for name in (
        "ctrl",
        "act",
        "qacc_warmstart",
        "qfrc_applied",
        "xfrc_applied",
        "mocap_pos",
        "mocap_quat",
        "userdata",
    ):
        np.testing.assert_array_equal(getattr(data, name), getattr(restored, name))
    for _ in range(20):
        mujoco.mj_step(model, data)
        mujoco.mj_step(model, restored)
        assert PhysicalCheckpoint.capture(model, data) == PhysicalCheckpoint.capture(
            model, restored
        )
