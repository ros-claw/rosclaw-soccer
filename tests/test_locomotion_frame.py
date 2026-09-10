import math

import pytest

from rosclaw_soccer.training.locomotion_frame import (
    LocomotionFrameConfig,
    select_locomotion_reflection,
    select_locomotion_reflection_batch,
)


def test_without_declared_idle_phase_legacy_sign_is_unchanged():
    for velocity in (-1.0, -0.02, -0.001, -1e-6, 0.0, 0.001, 1.0):
        assert select_locomotion_reflection(velocity, prefer_idle_frame=False) == (velocity < -1e-6)


def test_idle_phase_uses_trained_idle_convention_not_previous_mirrored_frame():
    for velocity in (-0.02, -0.001, 0.0, 0.02):
        assert not select_locomotion_reflection(velocity, prefer_idle_frame=True)
        assert select_locomotion_reflection(
            velocity,
            prefer_idle_frame=True,
            config=LocomotionFrameConfig(idle_reflected=True),
        )
    assert select_locomotion_reflection(-0.021, prefer_idle_frame=True)
    assert not select_locomotion_reflection(0.021, prefer_idle_frame=True)


@pytest.mark.parametrize("value", [True, -0.01, 0.11, math.nan, math.inf, "0.02"])
def test_bad_config_rejected(value):
    with pytest.raises(ValueError):
        LocomotionFrameConfig(idle_deadband_mps=value)


@pytest.mark.parametrize("value", [True, math.nan, math.inf, 10.1, "0"])
def test_bad_scalar_velocity_rejected(value):
    with pytest.raises(ValueError):
        select_locomotion_reflection(value, prefer_idle_frame=True)


def test_phase_and_config_are_not_truthiness_flags():
    with pytest.raises(ValueError):
        select_locomotion_reflection(0.0, prefer_idle_frame=1)
    with pytest.raises(ValueError):
        LocomotionFrameConfig(idle_reflected=1)
    with pytest.raises(ValueError):
        select_locomotion_reflection(0.0, prefer_idle_frame=True, config={})


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_batch_matches_scalar_and_leaves_inputs_unchanged(dtype):
    torch = pytest.importorskip("torch")
    velocity = torch.tensor([-0.7, -0.03, -0.01, -1e-6, 0, 0.01, 0.7], dtype=getattr(torch, dtype))
    idle = torch.tensor([False, True, True, False, True, False, True])
    before = velocity.clone()
    result = select_locomotion_reflection_batch(velocity, prefer_idle_frame=idle)
    expected = [
        select_locomotion_reflection(float(v), prefer_idle_frame=bool(i))
        for v, i in zip(velocity, idle, strict=True)
    ]
    assert result.tolist() == expected
    assert result.dtype == torch.bool and result.device == velocity.device
    assert torch.equal(before, velocity)


@pytest.mark.parametrize("bad", ["shape", "empty", "gradient", "integer", "nan", "phase_dtype"])
def test_batch_rejects_invalid_evidence(bad):
    torch = pytest.importorskip("torch")
    velocity = torch.zeros(2)
    idle = torch.zeros(2, dtype=torch.bool)
    if bad == "shape":
        velocity = torch.zeros(2, 1)
    elif bad == "empty":
        velocity = torch.zeros(0)
    elif bad == "gradient":
        velocity.requires_grad_(True)
    elif bad == "integer":
        velocity = torch.zeros(2, dtype=torch.int64)
    elif bad == "nan":
        velocity[0] = math.nan
    else:
        idle = torch.zeros(2)
    with pytest.raises(ValueError):
        select_locomotion_reflection_batch(velocity, prefer_idle_frame=idle)
