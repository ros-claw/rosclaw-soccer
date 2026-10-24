import numpy as np
import pytest
import torch
from test_joint_delivery_teacher import row
from test_stochastic_refinement_pilot import fixture

from rosclaw_soccer.rsi.trajectory_peer_baseline import peer_return_baselines
from scripts.rsi_refinement_onpolicy_data import episodes


def data(tmp_path):
    source, _ = fixture()
    n, tail = 3, 100
    for i, r in enumerate(source["records"]):
        r.update(
            stochastic_controls=n,
            motion_reference_controls=n,
            controls=n + tail,
            total_observed_controls=n + tail,
            outcome=row(0.1 + 0.02 * (i % 8), 2.7)["outcome"],
        )
        active = np.array([True] * n + [False] * tail)
        a = tmp_path / f"a{i}.npz"
        s = tmp_path / f"s{i}.npz"
        np.savez(
            a,
            NN_obs=np.zeros((n + tail, 160), np.float32),
            NN_action=np.zeros((n + tail, 29), np.float32),
            executed_action=np.zeros((n + tail, 29), np.float32),
            NN_reference_nominal=np.zeros((n + tail, 29), np.float32),
            standing=~active,
        )
        np.savez(s, active=active, logp=np.zeros(n + tail, np.float32))
        r["artifacts"] = dict(
            A=dict(path=a.name), sampling=dict(path=s.name), initial=dict(hash=f"initial{i // 8}")
        )
    return source


def test_peer_episode_advantages_derive_from_other_actual_returns(tmp_path):
    source = data(tmp_path)
    batch = episodes(tmp_path, source, torch.ones(29), advantage_mode="peer-loo-v2")
    baselines, advantage = peer_return_baselines(
        [e["returns"] for e in batch], [e["initial_state_hash"] for e in batch]
    )
    flat = torch.cat(advantage)
    for e, b, a in zip(batch, baselines, advantage, strict=True):
        assert torch.equal(e["peer_baseline"], b)
        assert torch.allclose(
            e["advantages"], (a - flat.mean()) / flat.std(unbiased=False).clamp_min(1e-6)
        )
        assert len(e["observations"]) == 3
        assert e["old_values"].eq(0).all()
    legacy = episodes(tmp_path, source, torch.ones(29))
    assert all("peer_baseline" not in e for e in legacy)
    assert not torch.allclose(batch[0]["advantages"], legacy[0]["advantages"])


def test_unbound_initial_group_cannot_sneak_into_peer_training(tmp_path):
    source = data(tmp_path)
    source["records"][0]["artifacts"]["initial"]["hash"] = "different"
    with pytest.raises(ValueError, match="physical initial"):
        episodes(tmp_path, source, torch.ones(29), advantage_mode="peer-loo-v2")


def test_full_coverage_data_uses_actual_recovery_rows_and_no_extra_delay(tmp_path):
    from rosclaw_soccer.rsi.refinement_onpolicy_learning import joint_delivery_reward

    source = data(tmp_path)
    for r in source["records"]:
        r.update(
            controller_kind="pure-refinement-full-recovery-stochastic-native-v2",
            stochastic_controls=103,
            deterministic_recovery_excluded_from_actor_rows=False,
        )
        np.savez(
            tmp_path / r["artifacts"]["sampling"]["path"],
            active=np.ones(103, np.bool_),
            logp=np.zeros(103, np.float32),
        )
    batch = episodes(tmp_path, source, torch.ones(29))
    assert all(len(e["observations"]) == 103 for e in batch)
    assert batch[0]["returns"][-1] == np.float32(
        joint_delivery_reward(source["records"][0]["outcome"])
    )
    # Merely relabelling the old inactive tail cannot manufacture actor evidence.
    np.savez(
        tmp_path / source["records"][0]["artifacts"]["sampling"]["path"],
        active=np.array([True] * 3 + [False] * 100),
        logp=np.zeros(103, np.float32),
    )
    with pytest.raises(ValueError, match="actually sampled"):
        episodes(tmp_path, source, torch.ones(29))
