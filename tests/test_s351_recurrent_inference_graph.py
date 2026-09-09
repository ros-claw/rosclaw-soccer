from types import SimpleNamespace

import pytest

from rosclaw_soccer.providers.g1.recurrent_inference_state import detach_frozen_recurrent_state


def test_detach_retains_memory_values_storage_and_forward_outputs():
    torch = pytest.importorskip("torch")
    torch.manual_seed(351)
    rnn = torch.nn.LSTM(3, 8)
    old = SimpleNamespace(hidden_state=torch.zeros(1, 1, 8), cell_state=torch.zeros(1, 1, 8))
    new = SimpleNamespace(hidden_state=old.hidden_state.clone(), cell_state=old.cell_state.clone())
    for observation in torch.randn(12, 1, 1, 3):
        expected, (old.hidden_state, old.cell_state) = rnn(
            observation, (old.hidden_state, old.cell_state)
        )
        actual, (new.hidden_state, new.cell_state) = rnn(
            observation, (new.hidden_state, new.cell_state)
        )
        assert torch.equal(actual, expected)
        addresses = new.hidden_state.data_ptr(), new.cell_state.data_ptr()
        detach_frozen_recurrent_state(new, frozen_inference=True)
        assert addresses == (new.hidden_state.data_ptr(), new.cell_state.data_ptr())
        assert new.hidden_state.grad_fn is None and new.cell_state.grad_fn is None
        assert torch.equal(new.hidden_state, old.hidden_state)
        assert torch.equal(new.cell_state, old.cell_state)
    assert all(parameter.grad is None for parameter in rnn.parameters())


@pytest.mark.parametrize("bad", ["missing", "nan", "shape", "dtype", "oversized"])
def test_invalid_pair_is_rejected_without_partial_replacement(bad):
    torch = pytest.importorskip("torch")
    hidden = torch.ones(1, 1, 8, requires_grad=True) * 2
    cell = torch.ones_like(hidden)
    if bad == "missing":
        cell = None
    if bad == "nan":
        cell[:] = float("nan")
    if bad == "shape":
        cell = torch.zeros(1, 2, 8)
    if bad == "dtype":
        cell = cell.double()
    if bad == "oversized":
        cell = torch.zeros(1, 1, 4097)
    policy = SimpleNamespace(hidden_state=hidden, cell_state=cell)
    with pytest.raises(ValueError):
        detach_frozen_recurrent_state(policy, frozen_inference=True)
    assert policy.hidden_state is hidden and policy.cell_state is cell


@pytest.mark.parametrize("owner", [False, 1, None])
def test_training_or_implicit_ownership_cannot_discard_gradients(owner):
    with pytest.raises(ValueError, match="frozen"):
        detach_frozen_recurrent_state(SimpleNamespace(), frozen_inference=owner)
