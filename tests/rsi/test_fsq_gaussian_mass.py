import pytest

from rosclaw_soccer.rsi.fsq_gaussian_mass import fsq_log_mass


def test_all32_bins_normalize_and_tail_gradients_stay_finite():
    import torch

    for center in (-2.0, -0.2, 0.0, 0.3, 2.0):
        mean = torch.full((32,), center, requires_grad=True)
        std = torch.full((32,), 0.02, requires_grad=True)
        token = torch.arange(-16, 16, dtype=torch.float32) / 16
        logp = fsq_log_mass(mean, std, token)
        assert torch.isfinite(logp).all()
        assert logp.exp().sum().item() == pytest.approx(1.0, abs=1e-6)
        logp.sum().backward()
        assert torch.isfinite(mean.grad).all() and torch.isfinite(std.grad).all()


def test_probability_matches_actual_sampler_and_bin_equivalent_raws_share_gradient():
    import torch

    torch.set_num_threads(1)
    generator = torch.Generator().manual_seed(7)
    raw = torch.randn(100000, generator=generator) * 0.02
    token = torch.round(15.515501022338867 * torch.tanh(raw + 0.03223699703812599) - 0.5) / 16
    symbols = torch.arange(-16, 16, dtype=torch.float32) / 16
    predicted = fsq_log_mass(torch.zeros(32), torch.full((32,), 0.02), symbols).exp()
    observed = torch.stack([(token == s).float().mean() for s in symbols])
    torch.testing.assert_close(predicted, observed, atol=0.003, rtol=0.02)
    mean = torch.tensor([0.001, 0.001], requires_grad=True)
    std = torch.full((2,), 0.02)
    equal_token = torch.zeros(2)
    p = fsq_log_mass(mean, std, equal_token)
    p.sum().backward()
    assert p[0] == p[1] and mean.grad[0] == mean.grad[1]


def test_non_grid_symbol_and_nonpositive_std_rejected():
    import torch

    with pytest.raises(ValueError):
        fsq_log_mass(torch.zeros(2), torch.ones(2), torch.full((2,), 0.01))
    with pytest.raises(ValueError):
        fsq_log_mass(torch.zeros(2), torch.zeros(2), torch.zeros(2))
