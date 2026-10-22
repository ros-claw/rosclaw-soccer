import numpy as np
import pytest
import torch

from rosclaw_soccer.rsi.paid_replay_neural import (
    ReplayMotorHead,
    replay_certificate,
    replay_config,
    training_arrays,
)


def fixture():
    x = np.random.default_rng(3).normal(size=(8, 288)).astype(np.float32)
    y = np.zeros((8, 29), np.float32)
    y[:4, 0] = 0.1
    return dict(
        features=x,
        target=y,
        parent_delta=np.zeros_like(y),
        active=np.array([True] * 4 + [False] * 4),
    )


def test_fixed_seeds_not_cherry_selected_and_no_cone_gate():
    configs = [replay_config(i) for i in range(3)]
    assert len({c["seed"] for c in configs}) == 3
    assert all(c["updates"] == 6144 and not c["geometric_support_gate"] for c in configs)
    assert all(
        c["primary_seed_index"] == 0 and not c["best_seed_selection_authorized"] for c in configs
    )
    with pytest.raises(ValueError):
        replay_config(True)


def test_initial_identity_and_final_authority_bound():
    net = ReplayMotorHead(torch.zeros(288), torch.ones(288))
    x = torch.randn(4, 288)
    prior = torch.full((4, 29), 0.2)
    assert torch.equal(net.combine(x, prior), prior)
    with torch.no_grad():
        net.net[-1].bias.fill_(100)
    assert net(x).abs().max() <= 0.4
    assert net.combine(x, prior).abs().max() <= 0.2


def test_real_gradient_updates_without_changing_normalization():
    torch.manual_seed(3)
    net = ReplayMotorHead(torch.zeros(288), torch.ones(288))
    before = net.feature_mean.clone()
    z = fixture()
    x, target, _, _ = training_arrays(z)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    initial = float((net(x) - target).square().mean().detach())
    for _ in range(8):
        loss = (net(x) - target).square().mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    assert float((net(x) - target).square().mean().detach()) < initial
    assert torch.equal(before, net.feature_mean)
    assert all(not v.requires_grad for v in (net.feature_mean, net.feature_scale))
    certificate = replay_certificate(net, z)
    assert certificate["CPU_batch1_rows"] == 8
    assert not certificate["exact_retention_claim"] and not certificate["physical_qualification"]


@pytest.mark.parametrize("bad", ["nan", "no_rehearsal", "hidden_label", "bound"])
def test_bad_data_fails_before_training(bad):
    z = fixture()
    if bad == "nan":
        z["features"][0, 0] = np.nan
    elif bad == "no_rehearsal":
        z["active"][:] = True
    elif bad == "hidden_label":
        z["target"][-1, 0] = 0.01
    else:
        z["target"][0, 0] = 0.5
    with pytest.raises(ValueError):
        training_arrays(z)
