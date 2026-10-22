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


def test_tail_replay_new_protocol_does_not_reinterpret_old_config():
    old = replay_config(0)
    new = replay_config(0, tail_replay=True)
    assert "tail_replay" not in old and old["protocol"].endswith(".v1")
    assert new["tail_replay"] and new["protocol"].endswith(".v2")
    assert new["updates"] == old["updates"] == 6144
    assert new["uniform_rehearsal_batch"] + new["hard_rehearsal_batch"] == old["rehearsal_batch"]
    with pytest.raises(ValueError):
        replay_config(0, tail_replay=1)


def test_tail_mining_and_gradients_execute_on_train_only_toy_data(monkeypatch):
    from rosclaw_soccer.rsi import paid_replay_neural as module

    cfg = replay_config(0, tail_replay=True)
    # Unit-only tiny loop exercises actual mining/backprop without pretending
    # to be the unmodified production6144-update protocol.
    cfg.update(
        updates=4,
        active_batch=4,
        rehearsal_batch=4,
        uniform_rehearsal_batch=2,
        hard_rehearsal_batch=2,
        hard_refresh_every=2,
        hard_pool_size=4,
        tail_top_rows=2,
    )
    monkeypatch.setattr(module, "replay_config", lambda index, **kwargs: cfg)
    progress = []
    _, _, counts, optimizer = module.fit_replay_head(
        fixture(), 0, "cpu", progress.append, tail_replay=True
    )
    assert counts["completed_updates"] == counts["nonzero_gradient_updates"] == 4
    assert counts["hard_pool_refreshes"] == 2
    assert counts["hard_pool_forward_rows"] == 2 * 4
    assert counts["uniform_rehearsal_forward_rows"] == counts["hard_rehearsal_forward_rows"] == 8
    assert all(float(s["step"]) == 4 for s in optimizer["state"].values())
    assert progress[0]["tail_loss"] == 0
