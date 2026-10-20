"""Pipeline control-flow fixtures; never new native or learning evidence."""

import json
from types import SimpleNamespace

import pytest

from scripts import rsi_online_kinematic_motion as online


def test_each_round_uses_new_policy_and_persistent_optimizer(tmp_path, monkeypatch):
    from torch import nn

    class Fake(nn.Module):
        def __init__(self):
            super().__init__()
            self.parent = nn.Linear(1, 1)
            self.adapter = nn.Linear(1, 1)

    fake = Fake()
    monkeypatch.setattr(online, "load_model", lambda *args: fake)
    calls, optimizer_ids = [], []

    def shards(a, folder, mode, path, weight_hash, indices, round_index):
        calls.append((mode, weight_hash, round_index, list(indices)))
        if mode == "collect":
            assert len(indices) == 32
            return []
        rows = [
            dict(
                course_index=i,
                score=dict(
                    qualified=i in (0, 10),
                    minimum_pelvis_z_m=0.75,
                    minimum_upright=0.99,
                    maximum_nonfoot_force_n=0.0,
                ),
                prefix_diagnostics=dict(
                    prefix_physical_safe=True, post_boundary_control_can_cause_first_touch=True
                ),
            )
            for i in indices
        ]
        return [(None, dict(result=dict(results=rows)))]

    def update(a, policy, optimizer, data, folder, weight_hash, round_index):
        assert policy is fake
        optimizer_ids.append(id(optimizer))
        return dict(checkpoint_hash=f"new{round_index}", behavior_checkpoint_hash=weight_hash)

    monkeypatch.setattr(online, "run_shards", shards)
    monkeypatch.setattr(online, "update_policy", update)
    a = SimpleNamespace(
        seed=17, policy=tmp_path / "initial.pt", expected_policy_hash="initial", output=tmp_path
    )
    bd = dict(
        rows=[
            dict(split="validation" if i in (7, 13, 20, 23, 25, 28, 30, 38) else "train")
            for i in range(40)
        ]
    )
    result = online.pipeline(a, {}, bd, {})
    assert len(set(optimizer_ids)) == 1
    assert [h for mode, h, _, _ in calls if mode == "collect"] == ["initial", "new0", "new1"]
    assert [h for mode, h, _, _ in calls if mode == "evaluate"] == ["new0", "new1", "new2"]
    assert result["new_native_steps"] == 3 * (32 + 40) * 6000
    assert result["optimizer_steps"] == 12 and result["fresh_exam_pass"] is False


def test_missing_collection_rejected_before_gradients(tmp_path):
    # No Torch model is called and no empty episode can be counted as32.
    with pytest.raises(ValueError, match="all32"):
        online.update_policy(None, None, None, [], tmp_path, "weights", 0)


def test_every_round_and_experiment_has_explicit_new_episode_noise():
    seeds = [
        online.episode_noise_seed(seed, round_index) + i
        for seed in (17, 29, 43)
        for round_index in range(3)
        for i in range(40)
    ]
    assert len(seeds) == len(set(seeds)) == 360
    with pytest.raises(ValueError):
        online.episode_noise_seed(True, 0)


def test_real_four_step_actor_critic_update_with_fixture_sequences(tmp_path, monkeypatch):
    import numpy as np
    import torch
    from torch import nn

    from scripts.rsi_native_context_recovery_common import digest

    class FakePolicy(nn.Module):
        def __init__(self):
            super().__init__()
            self.parent = nn.Linear(1, 1).requires_grad_(False)
            self.adapter = nn.Linear(108, 64)
            self.value_adapter = nn.Linear(108, 1)
            self.log_std = nn.Parameter(torch.full((64,), -3.9))

        def evaluate_raw(self, x, raw):
            mean = 0.1 * torch.tanh(self.adapter(x))
            std = self.log_std.exp().expand_as(mean)
            value = self.value_adapter(x).squeeze(-1)
            distribution = torch.distributions.Normal(mean, std)
            return (
                distribution.log_prob(raw).sum(-1),
                distribution.entropy().sum(-1),
                value,
                mean,
                std,
            )

    torch.set_num_threads(1)
    torch.manual_seed(42)
    policy = FakePolicy()
    monkeypatch.setattr(online, "load_model", lambda *args: FakePolicy())
    x = torch.randn(32, 300, 108) * 0.03
    with torch.no_grad():
        mean = 0.1 * torch.tanh(policy.adapter(x))
        std = policy.log_std.exp().expand_as(mean)
        raw = mean + std * torch.randn_like(mean)
        lp, _, value, _, _ = policy.evaluate_raw(x, raw)
    root = tmp_path / "collect"
    root.mkdir()
    rows = []
    for i in range(32):
        part = root / f"course{i:02d}"
        part.mkdir()
        np.savez(
            part / "native.npz", student_features=x[i].numpy(), value_estimate=value[i].numpy()
        )
        np.savez(
            part / "sampling.npz",
            raw=raw[i].numpy(),
            log_prob=lp[i].numpy(),
            rewards=torch.randn(300).numpy() * 0.01,
            mean=mean[i].numpy(),
            std=std[i].numpy(),
        )
        rows.append(
            dict(
                split="train",
                course_index=i,
                native_hash=digest(part / "native.npz"),
                sampling_hash=digest(part / "sampling.npz"),
            )
        )
    inputs, world = tmp_path / "inputs.json", tmp_path / "world"
    world.write_bytes(b"fixture, not native model")
    inputs.write_text(json.dumps(dict(experts=[dict(checkpoint_hash="fixture")], world=str(world))))
    output = tmp_path / "learn"
    output.mkdir()
    a = SimpleNamespace(inputs=inputs, policy=tmp_path / "unused.pt", expected_head="fixture")
    optimizer = torch.optim.Adam([p for p in policy.parameters() if p.requires_grad], lr=1e-4)
    result = online.update_policy(
        a, policy, optimizer, [(root, dict(result=dict(results=rows)))], output, "behavior", 0
    )
    assert result["persistent_adam_update_count"] == 4
    assert len(result["updates"]) == 4
    assert any(v > 0 for k, v in result["parameter_changes"].items() if k.startswith("adapter."))
    assert any(
        v > 0 for k, v in result["parameter_changes"].items() if k.startswith("value_adapter.")
    )
    assert all(v == 0 for k, v in result["parameter_changes"].items() if k.startswith("parent."))
    assert all(int(state["step"]) == 4 for state in optimizer.state.values())
