from dataclasses import dataclass

import pytest


def test_unqualified_kernel_rejected_before_data_access(monkeypatch):
    mjw = pytest.importorskip("mujoco_warp")
    pytest.importorskip("warp")
    from rosclaw_soccer.providers.g1.kinematic_snapshot import WarpKinematicsSnapshot

    def changed_kernel(model, data):
        raise AssertionError("must not execute an unqualified kernel")

    monkeypatch.setattr(mjw, "kinematics", changed_kernel)
    with pytest.raises(ValueError, match="unqualified"):
        WarpKinematicsSnapshot(None, None)


def test_qualified_kernel_still_requires_explicit_data():
    pytest.importorskip("mujoco_warp")
    pytest.importorskip("warp")
    from rosclaw_soccer.providers.g1.kinematic_snapshot import WarpKinematicsSnapshot

    with pytest.raises(ValueError, match="unqualified"):
        WarpKinematicsSnapshot(None, object())


def test_dataclass_type_is_not_a_live_simulation_instance():
    pytest.importorskip("mujoco_warp")
    pytest.importorskip("warp")
    from rosclaw_soccer.providers.g1.kinematic_snapshot import WarpKinematicsSnapshot

    @dataclass
    class Data:
        nworld: int = 1

    with pytest.raises(ValueError, match="unqualified"):
        WarpKinematicsSnapshot(None, Data)


@pytest.mark.parametrize("worlds", [0, 4097, True, 1.0, None])
def test_batch_bound_checked_before_cloning(worlds):
    pytest.importorskip("mujoco_warp")
    pytest.importorskip("warp")
    from rosclaw_soccer.providers.g1.kinematic_snapshot import WarpKinematicsSnapshot

    @dataclass
    class Data:
        nworld: object

    with pytest.raises(ValueError, match="bounded"):
        WarpKinematicsSnapshot(None, Data(worlds))


def test_actual_cpu_current_fk_private_output_and_no_live_solver_mutation():
    mujoco = pytest.importorskip("mujoco")
    np = pytest.importorskip("numpy")
    from rosclaw_soccer.providers.g1.kinematic_snapshot import CpuKinematicsSnapshot

    model = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><body><joint axis="0 0 1"/>'
        '<geom type="box" size=".1 .1 .1" pos=".4 0 0"/>'
        '</body></worldbody></mujoco>'
    )
    live = mujoco.MjData(model)
    live.qvel[0] = 2
    mujoco.mj_step(model, live)
    before = {
        key: getattr(live, key).copy()
        for key in ("qpos", "qvel", "ctrl", "qacc_warmstart", "qacc", "xpos", "xmat")
    }
    reader = CpuKinematicsSnapshot(model, [live])
    actual = reader.read()
    expected = mujoco.MjData(model)
    expected.qpos[:] = live.qpos
    mujoco.mj_forward(model, expected)
    np.testing.assert_array_equal(
        actual["xmat"][0].numpy(), expected.xmat.reshape(model.nbody, 3, 3)
    )
    assert not np.array_equal(actual["xmat"][0].numpy().reshape(model.nbody, 9), live.xmat)
    actual["qpos"].fill_(999)
    np.testing.assert_array_equal(reader.read()["qpos"][0].numpy(), live.qpos)
    for key, value in before.items():
        np.testing.assert_array_equal(value, getattr(live, key))


def test_cpu_invalid_input_is_latched_after_repair():
    mujoco = pytest.importorskip("mujoco")
    from rosclaw_soccer.providers.g1.kinematic_snapshot import CpuKinematicsSnapshot

    model = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><body><joint/><geom size=".1"/></body></worldbody></mujoco>'
    )
    live = mujoco.MjData(model)
    reader = CpuKinematicsSnapshot(model, [live])
    live.qpos[0] = float("nan")
    with pytest.raises(ValueError):
        reader.read()
    live.qpos[0] = 0
    with pytest.raises(RuntimeError, match="invalid"):
        reader.read()
