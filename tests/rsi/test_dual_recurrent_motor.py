import pytest
import torch
from torch import nn

from rosclaw_soccer.rsi.dual_recurrent_motor import DualRecurrentMotor
from rosclaw_soccer.rsi.paid_full_entry_student import FrozenFullEntryActor


class ToyBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("mean", torch.zeros(1, 160))
        self.register_buffer("denominator", torch.ones(1, 160))
        self.rnn = nn.LSTM(160, 128, num_layers=2)
        self.head = nn.Linear(128, 29)

    def forward_with_features(self, obs, state):
        hidden, after = self.rnn(obs, state)
        return self.head(hidden), after, hidden


class ToyMemory(nn.Module):
    def forward(self, features):
        return 0.02 * torch.tanh(features[:, :29])


def model():
    torch.manual_seed(45)
    parent = FrozenFullEntryActor(ToyBase(), ToyMemory(), torch.full((29,), 0.5)).eval()
    return parent, DualRecurrentMotor(parent).eval()


def test_initial_private_history_exactly_matches_frozen_parent():
    parent, learner = model()
    reference = (torch.zeros(2, 1, 128), torch.zeros(2, 1, 128))
    private = learner.initial_state()
    with torch.no_grad():
        for _ in range(12):
            obs = torch.randn(1, 1, 160)
            expected, reference = parent(obs, reference)
            actual, private = learner(obs, private)
            assert torch.equal(actual, expected)
            assert torch.equal(private[0][..., :128], reference[0])


def test_plastic_update_changes_recurrent_weights_not_frozen_memory():
    parent, learner = model()
    before = {k: v.clone() for k, v in parent.state_dict().items()}
    initial = learner.plastic.rnn.weight_ih_l0.detach().clone()
    optimizer = torch.optim.Adam(learner.plastic_parameters(), lr=0.001)
    output, _, _ = learner.forward_with_evidence(torch.randn(8, 1, 160), learner.initial_state())
    output.square().mean().backward()
    optimizer.step()
    assert not torch.equal(learner.plastic.rnn.weight_ih_l0, initial)
    assert all(torch.equal(v, before[k]) for k, v in parent.state_dict().items())
    assert all(p.grad is None and not p.requires_grad for p in parent.parameters())


def test_explicit_new_hidden_contract_rejects_borrowed128_state():
    _, learner = model()
    with pytest.raises(ValueError):
        learner(torch.zeros(1, 1, 160), (torch.zeros(2, 1, 128), torch.zeros(2, 1, 128)))


def test_reference_memory_does_not_consume_plastic_history():
    _, learner = model()
    state = learner.initial_state()
    _, _, baseline = learner.forward_with_evidence(torch.zeros(1, 1, 160), state)
    modified = tuple(t.clone() for t in state)
    modified[0][..., 128:] = 5.0
    _, _, changed = learner.forward_with_evidence(torch.zeros(1, 1, 160), modified)
    assert torch.equal(baseline["reference_mean"], changed["reference_mean"])


def test_mean_authority_remains_bounded_even_after_bad_weight_change():
    _, learner = model()
    with torch.no_grad():
        learner.plastic.head.bias.fill_(100.0)
    _, _, e = learner.forward_with_evidence(torch.zeros(1, 1, 160), learner.initial_state())
    assert (e["final_mean_delta_radians"].abs() <= 0.2).all()
