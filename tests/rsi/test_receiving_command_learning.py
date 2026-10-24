import pytest
import torch

from rosclaw_soccer.rsi.receiving_command_learning import fit_command_bc


def data():
    """Synthetic regression fixture, NOT actual receiving success evidence."""
    g = torch.Generator().manual_seed(23)
    x = torch.randn(96, 112, generator=g)
    y = torch.tensor([0.25, 0, 0]) + torch.tensor([0.75, 0.5, 1]) * torch.tanh(x[:, :3] * 0.3)
    training = torch.arange(96) < 72
    validation = ~training
    return x, y, training, validation


def test_real_optimizer_updates_all_six_tensors_and_safe_reload(tmp_path):
    x, y, tr, va = data()
    steps = []
    result = fit_command_bc(
        x, y, tr, va, seed=123, steps=100, on_step=lambda step, loss, norm: steps.append(step)
    )
    assert steps == list(range(1, 101))
    assert len(result["changed_parameter_tensors"]) == 6
    assert result["optimizer_updates"] == 100
    assert result["after"]["train_mse"] < result["before"]["train_mse"] * 0.1
    actor = result.pop("actor")
    cp = tmp_path / "synthetic.pt"
    torch.save(result | dict(actor=actor.state_dict()), cp)
    restored = torch.load(cp, map_location="cpu", weights_only=True)
    assert len(restored["optimizer"]["state"]) == 6
    assert restored["batch_rng"].device.type == "cpu"
    assert all(s["step"] == 100 for s in restored["optimizer"]["state"].values())


def test_validation_states_do_not_change_normalization_or_any_trained_parameter():
    x, y, tr, va = data()
    global_rng = torch.get_rng_state().clone()
    first = fit_command_bc(x, y, tr, va, seed=123, steps=4)
    altered = x.clone()
    altered[va] *= 1000
    second = fit_command_bc(altered, y, tr, va, seed=123, steps=4)
    assert torch.equal(torch.get_rng_state(), global_rng)
    assert torch.equal(first["actor"].observation_mean, x[tr].mean(0))
    for k, v in first["actor"].state_dict().items():
        assert torch.equal(v, second["actor"].state_dict()[k])


@pytest.mark.parametrize("fault", ["overlap", "empty", "nonfinite", "bounds", "steps", "grad"])
def test_invalid_fitting_bank_rejected_before_optimizer(fault):
    x, y, tr, va = data()
    n = 4
    if fault == "overlap":
        va[0] = True
    elif fault == "empty":
        tr[:] = False
    elif fault == "nonfinite":
        x[0, 0] = float("nan")
    elif fault == "bounds":
        y[0, 0] = 1.1
    elif fault == "steps":
        n = 401
    else:
        x.requires_grad_(True)
    with pytest.raises(ValueError):
        fit_command_bc(x, y, tr, va, seed=123, steps=n)
