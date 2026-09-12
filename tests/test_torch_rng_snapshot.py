"""Portable RNG snapshots fail closed on omissions and implicit GPU remapping."""

import pytest

from rosclaw_soccer.training.torch_rng_snapshot import capture_torch_rng, restore_torch_rng

torch = pytest.importorskip("torch")


def test_cpu_replay_without_accessing_cuda(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("CPU-only snapshot touched CUDA")

    monkeypatch.setattr(torch.cuda, "get_rng_state", forbidden)
    monkeypatch.setattr(torch.cuda, "set_rng_state", forbidden)
    original = capture_torch_rng()
    try:
        expected = torch.randn(17)
        restore_torch_rng(original)
        assert torch.equal(expected, torch.randn(17))
        assert original["cpu"].device.type == "cpu"
        assert original["cuda"] == {}
    finally:
        restore_torch_rng(original)


@pytest.mark.parametrize(
    "devices",
    [
        [],
        ("cuda",),
        ("cpu",),
        ("cuda:-1",),
        ("cuda:00",),
        ("cuda:1000",),
        (0,),
        ("cuda:0", "cuda:0"),
    ],
)
def test_reject_implicit_devices(devices):
    with pytest.raises(ValueError):
        capture_torch_rng(cuda_devices=devices)


@pytest.mark.parametrize("key", ["schema", "cpu", "cuda", "cuda_visible_devices"])
def test_missing_fields_leave_rng_unchanged(key):
    snapshot = capture_torch_rng()
    del snapshot[key]
    before = torch.get_rng_state().clone()
    with pytest.raises(ValueError):
        restore_torch_rng(snapshot)
    assert torch.equal(before, torch.get_rng_state())


def test_malformed_state_and_device_mapping_do_not_change_default_rng(monkeypatch):
    before = torch.get_rng_state().clone()
    snapshot = capture_torch_rng()
    snapshot["cpu"] = torch.zeros(3, dtype=torch.uint8)
    with pytest.raises(RuntimeError):
        restore_torch_rng(snapshot)
    assert torch.equal(before, torch.get_rng_state())
    snapshot = capture_torch_rng()
    snapshot["cuda"] = {"cuda:0": torch.zeros(16, dtype=torch.uint8)}
    snapshot["cuda_visible_devices"] = "3"
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")
    with pytest.raises(ValueError):
        restore_torch_rng(snapshot, cuda_devices=("cuda:0",))
    assert torch.equal(before, torch.get_rng_state())


@pytest.mark.parametrize(
    "state",
    [
        torch.zeros(8),
        torch.zeros(2, 2, dtype=torch.uint8),
        torch.zeros(0, dtype=torch.uint8),
        torch.zeros(16, dtype=torch.uint8)[::2],
    ],
)
def test_invalid_storage_rejected(state):
    snapshot = capture_torch_rng()
    snapshot["cpu"] = state
    with pytest.raises(ValueError):
        restore_torch_rng(snapshot)
