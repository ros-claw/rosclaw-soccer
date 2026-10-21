import numpy as np
import pytest
import torch

from rosclaw_soccer.rsi.paid_trainable_actor import PaidTrainableActor, onnx_to_torch_gates


@pytest.mark.parametrize("width", [None, 160, 128])
def test_gate_order_is_iofc_to_ifgo_not_identity(width):
    shape = (512,) if width is None else (512, width)
    a = np.broadcast_to(
        np.repeat(np.arange(4, dtype=np.float32), 128).reshape(
            (512,) + (() if width is None else (1,))
        ),
        shape,
    ).copy()
    b = onnx_to_torch_gates(a)
    assert b.dtype == torch.float32
    assert [float(x.flatten()[0]) for x in b.chunk(4)] == [0, 2, 3, 1]
    a[:] = 7
    assert float(b.flatten()[0]) == 0


@pytest.mark.parametrize(
    "a", [np.zeros(511, np.float32), np.zeros(512, np.float64), np.full(512, np.nan, np.float32)]
)
def test_gate_conversion_rejects_bad_arrays(a):
    with pytest.raises(ValueError):
        onnx_to_torch_gates(a)


def synthetic_actor():
    model = PaidTrainableActor.__new__(PaidTrainableActor)
    torch.nn.Module.__init__(model)
    model.register_buffer("mean", torch.zeros(1, 160))
    model.register_buffer("denominator", torch.ones(1, 160))
    model.rnn = torch.nn.LSTM(160, 128, num_layers=2)
    model.actor = torch.nn.Linear(128, 29)
    return model


def test_differentiable_full_history_and_independent_batch_players():
    torch.manual_seed(17)
    model = synthetic_actor()
    obs = torch.randn(5, 3, 160, requires_grad=True)
    state = (torch.zeros(2, 3, 128), torch.zeros(2, 3, 128))
    action, (h, c) = model(obs, state)
    for player in range(3):
        isolated, (hi, ci) = model(
            obs[:, player : player + 1],
            (state[0][:, player : player + 1], state[1][:, player : player + 1]),
        )
        torch.testing.assert_close(action[:, player : player + 1], isolated)
        torch.testing.assert_close(h[:, player : player + 1], hi)
        torch.testing.assert_close(c[:, player : player + 1], ci)
    action[-1].square().sum().backward()
    assert obs.grad is not None and torch.isfinite(obs.grad).all()
    assert obs.grad[0].abs().sum() > 0
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert not state[0].any() and not state[1].any()


def test_invalid_state_and_observation_rejected_without_padding():
    model = synthetic_actor()
    with pytest.raises(ValueError, match="160"):
        model(torch.zeros(1, 1, 159), (torch.zeros(2, 1, 128), torch.zeros(2, 1, 128)))
    with pytest.raises(ValueError, match="states"):
        model(torch.zeros(1, 2, 160), (torch.zeros(2, 1, 128), torch.zeros(2, 1, 128)))
