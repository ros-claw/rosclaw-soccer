import pytest

from rosclaw_soccer.providers.g1.contact_storage import validate_active_contact_storage

torch = pytest.importorskip("torch")


def legacy_accepts(normal, worlds, valid, count):
    return bool(torch.isfinite(normal[valid]).all()) and not bool(
        ((worlds[valid] < 0) | (worlds[valid] >= count)).any()
    )


@pytest.mark.parametrize("occupied", [0, 1, 256])
def test_valid_strided_storage_with_uninitialized_padding(occupied):
    storage = torch.full((512, 6), float("nan"))
    normal = storage[:, 0]
    normal[:occupied] = torch.arange(occupied, dtype=torch.float32) - 5
    worlds = torch.full((512,), -123, dtype=torch.int32)
    worlds[:occupied] = torch.arange(occupied, dtype=torch.int32) % 16
    valid = torch.arange(512) < occupied
    assert not normal.is_contiguous()
    assert legacy_accepts(normal, worlds, valid, 16)
    validate_active_contact_storage(normal, worlds, valid, environment_count=16)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_padded_storage_matches_legacy_value_acceptance(dtype):
    generator = torch.Generator().manual_seed(115401)
    for _ in range(100):
        normal = torch.randn(257, generator=generator, dtype=dtype)
        worlds = torch.randint(-2, 18, (257,), generator=generator, dtype=torch.int32)
        valid = torch.rand(257, generator=generator) < 0.2
        normal[torch.rand(257, generator=generator) < 0.01] = float("nan")
        before = tuple(value.clone() for value in (normal, worlds, valid))
        if legacy_accepts(normal, worlds, valid, 16):
            validate_active_contact_storage(normal, worlds, valid, environment_count=16)
        else:
            with pytest.raises(FloatingPointError, match="invalid physical contact"):
                validate_active_contact_storage(normal, worlds, valid, environment_count=16)
        for actual, expected in zip((normal, worlds, valid), before, strict=True):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)


@pytest.mark.parametrize("fault", [float("nan"), float("inf"), -float("inf")])
def test_invalid_occupied_force_is_rejected_but_padding_is_not_read_as_contact(fault):
    normal = torch.tensor([1.0, fault])
    worlds = torch.tensor([0, 0], dtype=torch.int32)
    validate_active_contact_storage(
        normal, worlds, torch.tensor([True, False]), environment_count=1
    )
    with pytest.raises(FloatingPointError):
        validate_active_contact_storage(
            normal, worlds, torch.tensor([True, True]), environment_count=1
        )


@pytest.mark.parametrize("world", [-1, 2, 999999])
def test_out_of_range_occupied_world_is_rejected(world):
    normal = torch.tensor([1.0, 2.0])
    worlds = torch.tensor([0, world], dtype=torch.int64)
    validate_active_contact_storage(
        normal, worlds, torch.tensor([True, False]), environment_count=2
    )
    with pytest.raises(FloatingPointError):
        validate_active_contact_storage(
            normal, worlds, torch.tensor([True, True]), environment_count=2
        )


def test_empty_storage_and_no_occupied_slots_match_legacy():
    for normal, worlds, valid in (
        (torch.empty(0), torch.empty(0, dtype=torch.int32), torch.empty(0, dtype=torch.bool)),
        (torch.tensor([float("nan")]), torch.tensor([-999]), torch.tensor([False])),
    ):
        assert legacy_accepts(normal, worlds, valid, 1)
        validate_active_contact_storage(normal, worlds, valid, environment_count=1)


@pytest.mark.parametrize("fault", ["shape", "force_dtype", "world_dtype", "mask_dtype", "count"])
def test_storage_contract_fails_closed(fault):
    normal = torch.ones(2)
    worlds = torch.zeros(2, dtype=torch.int32)
    valid = torch.ones(2, dtype=torch.bool)
    count = 1
    if fault == "shape":
        worlds = worlds[:1]
    elif fault == "force_dtype":
        normal = normal.int()
    elif fault == "world_dtype":
        worlds = worlds.float()
    elif fault == "mask_dtype":
        valid = valid.float()
    else:
        count = True
    with pytest.raises(ValueError):
        validate_active_contact_storage(normal, worlds, valid, environment_count=count)


@pytest.mark.parametrize("fault", ["force", "world"])
def test_sampler_still_latches_invalid_contact_evidence(monkeypatch, fault):
    import contextlib
    import sys
    from types import SimpleNamespace

    from rosclaw_soccer.providers.g1.vector_contact import G1VectorContactProbe, PhysicsSampleClock

    monkeypatch.setitem(
        sys.modules,
        "warp",
        SimpleNamespace(
            stream_from_torch=lambda stream: stream,
            ScopedStream=lambda stream: contextlib.nullcontext(),
        ),
    )
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device: None)
    probe = G1VectorContactProbe.__new__(G1VectorContactProbe)
    probe.clock = PhysicsSampleClock()
    probe.clock.begin(1, 0)
    probe.motor = SimpleNamespace(
        episode_resets=1,
        physics_steps=1,
        device="cpu",
        _ready=True,
        _model=None,
        _data=None,
        config=SimpleNamespace(environment_count=1),
        _mjw=SimpleNamespace(contact_force=lambda *args: None),
    )
    probe._torch = torch
    probe._indices = None
    probe._forces = None
    probe._force = torch.ones(2, 6)
    probe._world = torch.zeros(2, dtype=torch.int32)
    probe._slots = torch.arange(2)
    probe._count = torch.tensor([2])
    if fault == "force":
        probe._force[0, 0] = float("nan")
    else:
        probe._world[0] = 1
    with pytest.raises(FloatingPointError):
        probe.sample()
    assert not probe.clock.valid
    with pytest.raises(RuntimeError, match="skipped/repeated"):
        probe.sample()
