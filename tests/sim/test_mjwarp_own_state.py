"""Own-state transport unit tests; these are not GPU physics evidence."""

from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest

from rosclaw_soccer.sim.mjwarp_own_state import MjWarpOwnStateDiagnostic


class Array:
    def __init__(self, value):
        self.value = np.asarray(value)

    def numpy(self):
        return self.value

    def assign(self, value):
        self.value = value.copy()


@pytest.fixture
def bridge():
    instance = object.__new__(MjWarpOwnStateDiagnostic)
    instance._model = SimpleNamespace(nu=2)
    instance._gpu_model = SimpleNamespace()
    instance._host = SimpleNamespace(time=0.0, ctrl=np.array([0.25, -0.5]))
    for name in (
        "qpos",
        "qvel",
        "xpos",
        "xquat",
        "cvel",
        "cdof",
        "subtree_com",
        "actuator_force",
        "efc_force",
    ):
        setattr(instance._host, name, np.zeros(2))
    instance._gpu = SimpleNamespace(
        nacon=Array([0]),
        nefc=Array([0]),
        naconmax=8,
        njmax=16,
        ctrl=Array([[0.0, 0.0]]),
        overflow=Array([0]),
    )
    calls = []

    def export(host, model, gpu, *, world_id):
        calls.append(("export", world_id))
        host.qpos[:] = instance.physics_steps * 2
        host.qvel[:] = instance.physics_steps * 3
        host.time = instance.physics_steps * 0.002
        host.efc_force[:] = 7

    def step(model, gpu):
        calls.append(("gpu_step", gpu.ctrl.value.copy()))

    instance._backend = SimpleNamespace(get_data_into=export, step=step)
    instance._wp = SimpleNamespace(ScopedDevice=lambda device: nullcontext())
    instance._device = "cuda:0"
    instance._valid = True
    instance.physics_steps = 0
    instance._export()
    calls.clear()
    return instance, calls


def test_own_state_export_and_only_control_uploaded(bridge):
    instance, calls = bridge
    instance.step(instance._model, instance._host)
    assert calls[0][0] == "gpu_step"
    np.testing.assert_array_equal(calls[0][1], [[0.25, -0.5]])
    assert calls[1] == ("export", 0)
    np.testing.assert_array_equal(instance._host.qpos, [2, 2])
    np.testing.assert_array_equal(instance._host.qvel, [3, 3])
    np.testing.assert_array_equal(instance._host.efc_force, [7, 7])
    assert instance.physics_steps == 1
    assert instance.qualification == "UNQUALIFIED_GPU_OWN_STATE_DIAGNOSTIC"


@pytest.mark.parametrize("field", ["qpos", "qvel", "time"])
def test_cpu_state_overwrite_rejected_and_latched(bridge, field):
    instance, calls = bridge
    if field == "time":
        instance._host.time = 1.0
    else:
        getattr(instance._host, field)[0] = 1
    with pytest.raises(ValueError, match="overwritten"):
        instance.step(instance._model, instance._host)
    assert calls == []
    with pytest.raises(RuntimeError, match="invalid"):
        instance.step(instance._model, instance._host)


@pytest.mark.parametrize("field, count", [("nacon", 9), ("nefc", 17), ("nacon", -1)])
def test_overflow_rejected_before_export(bridge, field, count):
    instance, calls = bridge
    setattr(instance._gpu, field, Array([count]))
    with pytest.raises(FloatingPointError, match="no truncated export"):
        instance.step(instance._model, instance._host)
    assert len(calls) == 1  # GPU stepped; overflowing state never exported.
    assert not instance._valid


@pytest.mark.parametrize("flag", [1 << bit for bit in range(31)])
def test_any_backend_overflow_flag_rejected_even_with_small_counts(bridge, flag):
    instance, calls = bridge
    instance._gpu.overflow = Array([flag])
    # Contact and constraint counts remain zero. Counts alone cannot detect
    # broadphase, CCD, line-search or main-solver failures.
    with pytest.raises(FloatingPointError, match="overflow/convergence"):
        instance.step(instance._model, instance._host)
    assert len(calls) == 1 and calls[0][0] == "gpu_step"
    assert not instance._valid
    assert instance._gpu.overflow.numpy()[0] == flag  # Never clear evidence.
    with pytest.raises(RuntimeError, match="invalid"):
        instance.step(instance._model, instance._host)


@pytest.mark.parametrize("value", [[], [[0]], [0, 0], [0.0], [True], [-1], [float("nan")]])
def test_malformed_overflow_field_rejected_before_export(bridge, value):
    instance, calls = bridge
    instance._gpu.overflow = Array(value)
    with pytest.raises(FloatingPointError, match="bitmask"):
        instance.step(instance._model, instance._host)
    assert len(calls) == 1
    assert not instance._valid


def test_missing_backend_overflow_capability_rejected(bridge):
    instance, calls = bridge
    del instance._gpu.overflow
    with pytest.raises(FloatingPointError, match="complete GPU overflow"):
        instance.step(instance._model, instance._host)
    assert len(calls) == 1
    assert not instance._valid


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_control_never_steps(bridge, value):
    instance, calls = bridge
    instance._host.ctrl[0] = value
    with pytest.raises(ValueError, match="control"):
        instance.step(instance._model, instance._host)
    assert not calls


def test_wrong_episode_rejected(bridge):
    instance, calls = bridge
    with pytest.raises(ValueError, match="bound"):
        instance.step(SimpleNamespace(nu=2), instance._host)
    assert not calls


def test_float32_overflow_control_never_steps(bridge):
    instance, calls = bridge
    instance._host.ctrl[0] = 1e300
    with pytest.raises(ValueError, match="float32"):
        instance.step(instance._model, instance._host)
    assert not calls
    assert not instance._valid


def test_export_failure_latches_without_retry(bridge):
    instance, calls = bridge
    instance._host.cvel[0] = float("nan")
    with pytest.raises(FloatingPointError, match="cvel"):
        instance.step(instance._model, instance._host)
    assert instance.physics_steps == 1
    assert not instance._valid
    with pytest.raises(RuntimeError):
        instance.step(instance._model, instance._host)
    assert len(calls) == 2


@pytest.mark.parametrize("device", ["cpu", "cuda", "cuda:-1", None])
def test_device_preflight_without_backend_import(device):
    with pytest.raises(ValueError, match="CUDA"):
        MjWarpOwnStateDiagnostic(None, None, device=device)


@pytest.mark.parametrize("capacity", [0, -1, True, 1.5])
def test_capacity_preflight_without_backend_import(capacity):
    with pytest.raises(ValueError, match="capacities"):
        MjWarpOwnStateDiagnostic(None, None, device="cuda:0", njmax=capacity)
