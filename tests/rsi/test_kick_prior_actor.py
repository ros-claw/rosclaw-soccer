import numpy as np
import pytest

torch = pytest.importorskip("torch")

from rosclaw_soccer.rsi.kick_prior_actor import KickPriorActor, TorchKickSession  # noqa: E402


def tensors():
    rng = np.random.default_rng(17)
    result = {}
    sizes = (547, 512, 256, 128, 29)
    for layer, (before, after) in enumerate(zip(sizes[:-1], sizes[1:], strict=True)):
        result[f"actor.{layer * 2}.weight"] = (rng.normal(size=(after, before)) * 0.01).astype(
            np.float32
        )
        result[f"actor.{layer * 2}.bias"] = np.zeros(after, np.float32)
    return result


def test_exact_tensor_copy_and_differentiability():
    torch.set_num_threads(1)
    weights = tensors()
    actor = KickPriorActor(weights)
    x = torch.ones((3, 547), dtype=torch.float32, requires_grad=True)
    result = actor(x)
    assert result.shape == (3, 29)
    np.testing.assert_array_equal(actor.actor[0].weight.detach().numpy(), weights["actor.0.weight"])
    result.square().sum().backward()
    assert bool(torch.isfinite(x.grad).all())
    assert all(p.grad is not None and bool(p.grad.abs().sum() > 0) for p in actor.parameters())
    assert actor.simulation_only and not actor.hardware_authorized
    with pytest.raises(ValueError):
        actor(torch.zeros((1, 549)))
    with pytest.raises(ValueError):
        actor(torch.full((1, 547), float("nan")))


def test_rejects_shape_drift_and_unknown_artifact(tmp_path):
    values = tensors()
    values["actor.0.bias"] = values["actor.0.bias"].astype(np.float64)
    with pytest.raises(ValueError):
        KickPriorActor(values)
    with pytest.raises(ValueError):
        KickPriorActor.from_onnx(tmp_path / "missing.onnx", expected_hash="unqualified")


def test_sim_inference_session_contract_and_private_call_count():
    session = TorchKickSession(KickPriorActor(tensors()))
    x = np.zeros((1, 547), np.float32)
    result = session.run(["actions"], {"obs": x})
    assert len(result) == 1 and result[0].shape == (1, 29)
    assert session.calls == 1
    with pytest.raises(ValueError):
        session.run(["torque"], {"obs": x})
    with pytest.raises(ValueError):
        session.run(["actions"], {"obs": np.zeros((2, 547), np.float32)})
    assert session.calls == 1
