import numpy as np
import pytest

from rosclaw_soccer.rsi.motion_capability_router import motion_routing_features


def test_routing_inputs_are_only_current_body_and_relative_ball():
    x = np.zeros(108, dtype=np.float32)
    x[1], x[2] = 0.75, 1.0
    x[70:76] = np.arange(6, dtype=np.float32)
    y = motion_routing_features(x)
    np.testing.assert_array_equal(y, np.asarray([0.75, 1, 0, 1, 2, 3, 4, 5], dtype=np.float32))
    x[12:70] = 50
    x[79:] = 300
    np.testing.assert_array_equal(motion_routing_features(x), y)
    x[2] = 0
    with pytest.raises(ValueError):
        motion_routing_features(x)


@pytest.mark.parametrize("confident,expected_owner", [(True, 1), (False, 0)])
def test_single_owner_uses_its_own_warmed_memory_and_replays_exactly(confident, expected_owner):
    import torch
    from torch import nn

    from scripts.rsi_frozen_motion_portfolio import FrozenMotionPortfolio

    class Expert(nn.Module):
        def __init__(self, i):
            super().__init__()
            self.i, self.calls = i, 0

        def sample(self, x, hidden, *, generator, stochastic):
            assert stochastic is False
            self.calls += 1
            h = torch.zeros(1, 1, 128) if hidden is None else hidden
            return dict(
                token=torch.full((1, 1, 64), self.i / 16.0),
                value=torch.zeros(1, 1),
                hidden=h + self.i + 1,
            )

    class Router(nn.Module):
        expert_count = 2

        def forward(self, x):
            return torch.tensor([[0.0, 10.0 if confident else 0.1]]), torch.zeros(1, 2)

    experts = [Expert(i).eval().requires_grad_(False) for i in range(2)]
    portfolio = FrozenMotionPortfolio(Router().eval().requires_grad_(False), experts)
    generator = torch.Generator().manual_seed(8)
    rng = generator.get_state().clone()
    outputs = []
    for _repeat in range(2):
        hidden = None
        rows = []
        for t in range(300):
            x = torch.zeros(1, 1, 108)
            x[0, 0, 0] = (300 - t) / 300
            x[0, 0, 1], x[0, 0, 2] = 0.75, 1.0
            row = portfolio.sample(x, hidden, generator=generator, stochastic=False)
            hidden = row["hidden"]
            rows.append(hidden.clone())
            if t == 20:
                assert float(hidden[0, 0, 0]) == (42.0 if expected_owner else 21.0)
        outputs.append(torch.stack(rows))
    assert torch.equal(outputs[0], outputs[1]) and torch.equal(rng, generator.get_state())
    assert [e.calls for e in experts] == ([40, 600] if expected_owner else [600, 40])
    assert portfolio.network_call_counts() == dict(
        policy_invocations=300, intention_network_calls=320, router_network_calls=1
    )
    assert all(
        r["selected_owner"] == expected_owner and r["router_calls"] == 1 for r in portfolio.audits
    )
    assert (
        portfolio.audits[0]["prefix_all_expert_memory_sha256"]
        == portfolio.audits[1]["prefix_all_expert_memory_sha256"]
    )
    with pytest.raises(ValueError):
        portfolio.sample(x, hidden, generator=generator, stochastic=True)
