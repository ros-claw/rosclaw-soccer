import numpy as np
import pytest

torch = pytest.importorskip("torch")

from rosclaw_soccer.rsi.velocity_prior_actor import VelocityPriorActor, phase_table  # noqa: E402


def test_phase_lookup_preserves_repeated_float32_addition():
    from rosclaw_soccer.providers.g1.velocity_policy import phase_delta

    values = phase_table(300)
    phase = phase_delta()
    for value in values:
        phase = np.float32(np.fmod(np.float32(phase + phase_delta()), np.float32(1)))
        assert value == phase
    with pytest.raises(ValueError, match="bounded"):
        phase_table(0)


def tensors():
    rng = np.random.default_rng(91)
    result = {
        "obs_normalizer._mean": rng.normal(size=(1, 98)).astype(np.float32),
        "onnx::Div_24": np.full((1, 98), 1.2, dtype=np.float32),
    }
    for index, (out, width) in zip(
        (0, 2, 4, 6), ((512, 98), (256, 512), (128, 256), (29, 128)), strict=True
    ):
        result[f"mlp.{index}.weight"] = (rng.normal(size=(out, width)) * 0.02).astype(np.float32)
        result[f"mlp.{index}.bias"] = (rng.normal(size=out) * 0.01).astype(np.float32)
    return result


def test_prior_output_and_new_reference_gradients():
    original = tensors()
    actor = VelocityPriorActor(original)
    observation = torch.randn(3, 168)
    actual = actor(observation)
    expected = (
        observation[:, :98] - torch.from_numpy(original["obs_normalizer._mean"])
    ) / torch.from_numpy(original["onnx::Div_24"])
    for index in (0, 2, 4, 6):
        expected = torch.nn.functional.linear(
            expected,
            torch.from_numpy(original[f"mlp.{index}.weight"]),
            torch.from_numpy(original[f"mlp.{index}.bias"]),
        )
        if index != 6:
            expected = torch.nn.functional.elu(expected)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=0)
    alternative = observation.clone()
    alternative[:, 98:] += 20
    assert torch.equal(actual, actor(alternative))
    actual.square().sum().backward()
    assert actor.mlp[0].weight.grad[:, 98:].abs().max() > 0
    assert not actor.prior_mean.requires_grad
    assert not actor.prior_denominator.requires_grad
    # Own all source arrays, not aliases to a caller's mutable numpy storage.
    original["mlp.6.bias"][:] += 100
    assert torch.equal(actual, actor(observation))


def test_invalid_prior_and_observation_fail_closed():
    bad = tensors()
    bad["onnx::Div_24"][0, 0] = 0
    with pytest.raises(ValueError, match="positive"):
        VelocityPriorActor(bad)
    actor = VelocityPriorActor(tensors())
    with pytest.raises(ValueError, match="reference features"):
        actor(torch.zeros(1, 98))
    with pytest.raises(ValueError, match="finite float32"):
        actor(torch.full((1, 168), float("nan")))
    with pytest.raises(ValueError, match="finite float32"):
        actor(torch.zeros(1, 168, dtype=torch.float64))
