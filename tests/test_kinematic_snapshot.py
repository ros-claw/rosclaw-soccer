import hashlib
import inspect
from dataclasses import dataclass
from pathlib import Path

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


def test_source_mismatch_still_rejected_before_cloning(monkeypatch):
    mjw = pytest.importorskip("mujoco_warp")
    wp = pytest.importorskip("warp")
    from rosclaw_soccer.providers.g1 import kinematic_snapshot as snapshot

    def forbidden_kernel(model, data):
        raise AssertionError("unqualified source must not execute")

    def forbidden_clone(value):
        raise AssertionError("unqualified source must not allocate")

    monkeypatch.setattr(mjw, "kinematics", forbidden_kernel)
    monkeypatch.setattr(wp, "clone", forbidden_clone)
    monkeypatch.setattr(
        snapshot,
        "_KINEMATICS_HASH",
        hashlib.sha256(inspect.getsource(forbidden_kernel).encode()).hexdigest(),
    )
    monkeypatch.setattr(snapshot, "_SMOOTH_SOURCE_HASH", "0" * 64)

    @dataclass
    class Data:
        nworld: int = 1

    with pytest.raises(ValueError, match="unqualified simulation kinematics kernel"):
        snapshot.WarpKinematicsSnapshot(None, Data())


@pytest.mark.parametrize("worlds", [0, 4097, True, 1.0, None])
def test_batch_bound_checked_before_cloning(worlds, monkeypatch):
    mjw = pytest.importorskip("mujoco_warp")
    wp = pytest.importorskip("warp")
    from rosclaw_soccer.providers.g1 import kinematic_snapshot as snapshot

    # Isolate batch validation from the locally installed kernel version. This
    # fake must never run or clone storage; production qualification is unchanged.
    def forbidden_kernel(model, data):
        raise AssertionError("batch rejection must precede execution")

    def forbidden_clone(value):
        raise AssertionError("batch rejection must precede allocation")

    monkeypatch.setattr(mjw, "kinematics", forbidden_kernel)
    monkeypatch.setattr(wp, "clone", forbidden_clone)
    monkeypatch.setattr(
        snapshot,
        "_KINEMATICS_HASH",
        hashlib.sha256(inspect.getsource(forbidden_kernel).encode()).hexdigest(),
    )
    monkeypatch.setattr(
        snapshot, "_SMOOTH_SOURCE_HASH", hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    )

    @dataclass
    class Data:
        nworld: object

    with pytest.raises(ValueError, match="bounded"):
        snapshot.WarpKinematicsSnapshot(None, Data(worlds))


def test_actual_cpu_current_fk_private_output_and_no_live_solver_mutation():
    mujoco = pytest.importorskip("mujoco")
    np = pytest.importorskip("numpy")
    from rosclaw_soccer.providers.g1.kinematic_snapshot import CpuKinematicsSnapshot

    model = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><body><joint axis="0 0 1"/>'
        '<geom type="box" size=".1 .1 .1" pos=".4 0 0"/>'
        "</body></worldbody></mujoco>"
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
