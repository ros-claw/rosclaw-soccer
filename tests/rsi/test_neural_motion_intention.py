import pytest

from rosclaw_soccer.rsi.neural_motion_intention import create_motion_intention_student


def test_quantized64_model_is_causal_recurrent_and_has_real_gradients():
    import torch

    torch.manual_seed(17)
    m = create_motion_intention_student()
    x = torch.randn(2, 30, 108)
    token, value, h = m(x)
    assert token.shape == (2, 30, 64) and h.shape == (1, 2, 128)
    assert token.min() >= -1 and token.max() <= 0.9375
    loss = (token - 0.125).square().mean() + 0.1 * value.square().mean()
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
    assert m.actor.weight.grad.abs().sum() > 0
    m.eval()
    with torch.no_grad():
        full, _, _ = m(x)
        hidden = None
        sequence = []
        for frame in range(30):
            a, _, hidden = m(x[:, frame : frame + 1], hidden)
            sequence.append(a)
    assert torch.equal(full * 16, torch.round(full * 16))
    torch.testing.assert_close(torch.cat(sequence, dim=1), full, rtol=0, atol=0)


def test_training_mean_is_explicit_and_preserves_fsq_lattice():
    import torch

    m = create_motion_intention_student()
    m.initialize_training_mean(torch.full((64,), -0.25))
    m.eval()
    with torch.no_grad():
        token, _, _ = m(torch.zeros(1, 1, 108))
    torch.testing.assert_close(token, torch.full((1, 1, 64), -0.25), rtol=0, atol=0)
    with pytest.raises(ValueError):
        m.initialize_training_mean(torch.ones(64))


@pytest.mark.parametrize("bad", ["nan", "double", "shape", "normalizer", "hidden"])
def test_invalid_inputs_never_invent_motion_tokens(bad):
    import torch

    m = create_motion_intention_student()
    x, hidden = torch.zeros(1, 2, 108), None
    if bad == "nan":
        x[0, 0, 0] = float("nan")
    if bad == "double":
        x = x.double()
    if bad == "shape":
        x = x[:, :, :107]
    if bad == "normalizer":
        m.observation_std[0] = 0
    if bad == "hidden":
        hidden = torch.zeros(1, 1, 129)
    with pytest.raises(ValueError):
        m(x, hidden)
