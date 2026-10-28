import torch

from scripts.rsi_train_receive_teacher_torque_bc import ResidualActor, metrics


def test_actor_outputs_bounded_29_joint_residual_and_metrics_distinguish_zeros():
    torch.manual_seed(1)
    actor = ResidualActor()
    x = torch.randn(4, 140)
    y = torch.zeros(4, 29)
    y[0, 0] = 4
    output = actor(x)
    assert output.shape == (4, 29) and torch.isfinite(output).all()
    assert (output.abs() <= 20).all()
    report = metrics(torch.zeros_like(y), y)
    assert report["active_samples"] == 1
    assert report["mse_nm2"] == report["zero_baseline_mse_nm2"]
    assert report["active_mse_nm2"] == report["active_zero_baseline_mse_nm2"]
    (output - y).square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in actor.parameters())
