from pathlib import Path

import numpy as np
import pytest

from rosclaw_soccer.growth.near_ball_residual import NearBallResidualPolicy, bounded_residual
from rosclaw_soccer.training.near_ball_residual_ppo import episodic_gae, update_private_actors


def policy() -> NearBallResidualPolicy:
    return NearBallResidualPolicy.initialize(
        tuple(f"agent.{i}" for i in range(8)), "sha256:" + "a" * 64
    )


def test_zero_actor_and_immutable_checkpoint(tmp_path: Path) -> None:
    p = policy()
    x = np.ones((8, 56))
    assert not p.act(x, np.random.default_rng(0), explore=False)[0].any()
    original = p.policy_hash
    with pytest.raises(ValueError):
        p.weights["w1"][0, 0, 0] = 99
    with pytest.raises(ValueError):
        p.weights["w1"].setflags(write=True)
    path = tmp_path / "actor.npz"
    p.save(path)
    assert NearBallResidualPolicy.load(path).policy_hash == original
    with pytest.raises(FileExistsError):
        p.save(path)
    with pytest.raises(ValueError):
        p.save(tmp_path / "actor")


def test_filter_and_seed_replay() -> None:
    p = policy()
    a = p.act(np.ones((8, 56)), np.random.default_rng(7), explore=True)[0]
    np.testing.assert_array_equal(
        a, p.act(np.ones((8, 56)), np.random.default_rng(7), explore=True)[0]
    )
    previous = np.zeros((8, 12))
    for _ in range(30):
        next_value = bounded_residual(np.ones((8, 12)) * 100, previous, np.ones(8, dtype=bool))
        assert np.max(np.abs(next_value - previous)) <= 0.02000000001
        assert np.max(np.abs(next_value)) <= 0.1
        previous = next_value
    assert np.max(bounded_residual(a, previous, np.zeros(8, dtype=bool))) < np.max(previous)
    with pytest.raises(ValueError):
        bounded_residual(a * np.nan, previous, np.ones(8, dtype=bool))


def test_gae_has_terminal_zero() -> None:
    adv, returns = episodic_gae(np.ones((2, 8)), np.zeros((2, 8)))
    np.testing.assert_allclose(adv[-1], 1)
    np.testing.assert_allclose(returns[0], 1 + 0.99 * 0.95)


def samples(p: NearBallResidualPolicy) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(12)
    n = 40
    obs = rng.normal(0, 0.1, (n, 8, 56))
    action, logp, values = zip(*(p.act(x, rng, explore=True) for x in obs), strict=True)
    t = {
        "time": np.arange(n) * 0.02,
        "residual_observations": obs,
        "residual_latent": np.asarray(action),
        "residual_log_probability": np.asarray(logp),
        "residual_value": np.asarray(values),
        "residual_active": np.zeros((n, 8), dtype=bool),
        "residual_applied": np.zeros((n, 8, 12)),
        "ball_pose": np.zeros((n, 7)),
        "ball_velocity": np.zeros((n, 6)),
    }
    for key in (
        "ball_contact_agent_code",
        "ball_contact_effector_code",
        "ball_nonfoot_contact_agent_code",
        "robot_robot_contact_first_code",
        "robot_robot_contact_second_code",
        "pass_source_agent_code",
        "pass_target_agent_code",
        "ball_contact_force_n",
    ):
        t[key] = np.zeros(n)
    t["residual_active"][:, 0] = True
    for agent in p.agent_ids:
        for suffix in ("_left_foot_position", "_right_foot_position", "_target_position"):
            t[agent.replace(".", "_") + suffix] = rng.normal(0, 0.1, (n, 3))
        t[agent.replace(".", "_") + "_pelvis_pose"] = np.ones((n, 7))
    return t


def test_actual_ppo_updates_only_sampled_private_actor() -> None:
    pytest.importorskip("torch")
    p = policy()
    child, rows = update_private_actors(p, [samples(p)])
    assert child.parent_hash == p.policy_hash
    assert rows[0]["updated"] and not any(r["updated"] for r in rows[1:])
    assert np.any(child.weights["w2"][0] != p.weights["w2"][0])
    for k in p.weights:
        np.testing.assert_array_equal(child.weights[k][1:], p.weights[k][1:])


def test_training_scope_freezes_sampled_roles_without_relabeling_rollouts():
    pytest.importorskip("torch")
    p = policy()
    trace = samples(p)
    trace["residual_active"][:, 1] = True
    original = {k: v.copy() for k, v in trace.items()}
    child, rows = update_private_actors(p, [trace], trainable_agent_ids=("agent.0",))
    replay, replay_rows = update_private_actors(p, [trace], trainable_agent_ids=("agent.0",))
    assert child.policy_hash == replay.policy_hash and rows == replay_rows
    assert rows[0]["updated"]
    assert rows[1]["active_samples"] == 40 and rows[1]["frozen_by_training_scope"]
    assert not any(r["updated"] for r in rows[1:])
    for k in p.weights:
        np.testing.assert_array_equal(child.weights[k][1:], p.weights[k][1:])
    for k in trace:
        np.testing.assert_array_equal(trace[k], original[k])
    all_roles, all_rows = update_private_actors(p, [trace], trainable_agent_ids=p.agent_ids)
    default, _ = update_private_actors(p, [trace])
    assert all_roles.policy_hash == default.policy_hash and all_rows[1]["updated"]
    assert rows[0]["core_plasticity"] != all_rows[0]["core_plasticity"]


@pytest.mark.parametrize(
    "scope", [(), [], ("agent.1", "agent.0"), ("agent.0", "agent.0"), ("unknown",), (True,)]
)
def test_private_ppo_rejects_ambiguous_or_unknown_training_scope(scope):
    pytest.importorskip("torch")
    p = policy()
    with pytest.raises(ValueError, match="trainable roles"):
        update_private_actors(p, [samples(p)], trainable_agent_ids=scope)


def test_private_ppo_binds_frozen_whole_body_components_without_changing_update():
    pytest.importorskip("torch")
    p = policy()
    trace = samples(p)
    plain, _ = update_private_actors(p, [trace])
    frozen = {"motor.red.finisher": "sha256:" + "b" * 64, "foundation.sonic": "sha256:" + "c" * 64}
    bound, rows = update_private_actors(p, [trace], frozen_policy_hashes=frozen)
    for key in p.weights:
        np.testing.assert_array_equal(bound.weights[key], plain.weights[key])
    bindings = {b["agent_id"]: b for b in rows[0]["core_plasticity"]["lease"]["bindings"]}
    for key, digest in frozen.items():
        assert bindings[key]["mode"] == "FROZEN"
        assert bindings[key]["policy_hash"] == digest
    for invalid in (
        {p.agent_ids[0]: "sha256:" + "b" * 64},
        {"motor": "unbound"},
        {1: "sha256:" + "b" * 64},
    ):
        with pytest.raises(ValueError):
            update_private_actors(p, [trace], frozen_policy_hashes=invalid)


def test_wrong_parent_rollout_rejected() -> None:
    pytest.importorskip("torch")
    p = policy()
    trace = samples(p)
    trace["residual_log_probability"][0, 0] += 0.01
    with pytest.raises(ValueError, match="frozen parent"):
        update_private_actors(p, [trace])


def test_policy_rejects_nonfinite_or_excessive_weights() -> None:
    p = policy()
    for bad in (float("nan"), float("inf"), 21.0):
        weights = {k: v.copy() for k, v in p.weights.items()}
        weights["w2"][0, 0, 0] = bad
        with pytest.raises(ValueError):
            NearBallResidualPolicy(p.agent_ids, p.body_hash, 0, p.parent_hash, weights)


def test_floating_generation_is_not_silently_truncated(tmp_path: Path) -> None:
    p = policy()
    path = tmp_path / "malformed.npz"
    np.savez_compressed(
        path,
        agent_ids=p.agent_ids,
        body_hash=p.body_hash,
        generation=1.5,
        parent_hash=p.parent_hash,
        **p.weights,
    )
    with pytest.raises(ValueError, match="integer scalar"):
        NearBallResidualPolicy.load(path)


def test_no_samples_means_no_weight_updates() -> None:
    pytest.importorskip("torch")
    p = policy()
    trace = samples(p)
    trace["residual_active"][:] = False
    child, rows = update_private_actors(p, [trace])
    assert not any(row["updated"] for row in rows)
    for key in p.weights:
        np.testing.assert_array_equal(child.weights[key], p.weights[key])


def test_behavior_rehearsal_reduces_distribution_drift_and_binds_immutable_reference():
    torch = pytest.importorskip("torch")
    from rosclaw_soccer.training.role_behavior_anchor import RoleBehaviorAnchor

    p = policy()
    trace = samples(p)
    anchor = RoleBehaviorAnchor(
        p, trace["residual_observations"], trace["residual_active"], "sha256:" + "e" * 64, 1000.0
    )
    plain, _ = update_private_actors(p, [trace], epochs=16)
    child, rows = update_private_actors(p, [trace], epochs=16, behavior_anchor=anchor)
    replay, replay_rows = update_private_actors(p, [trace], epochs=16, behavior_anchor=anchor)
    assert child.policy_hash == replay.policy_hash and rows == replay_rows

    def kl(model):
        params = {k: torch.tensor(v[0].copy()) for k, v in model.weights.items()}
        return float(anchor.kl_loss(params, 0))

    assert kl(child) < kl(plain)
    assert rows[0]["behavior_anchor"]["anchor_hash"] == anchor.anchor_hash
    assert rows[0]["behavior_anchor"]["kl_after"] == pytest.approx(kl(child))
    for key in p.weights:
        np.testing.assert_array_equal(child.weights[key][1:], p.weights[key][1:])
    with pytest.raises(ValueError, match="anchor"):
        update_private_actors(
            p,
            [trace],
            behavior_anchor=anchor,
            frozen_policy_hashes={"retention.anchor": "sha256:" + "f" * 64},
        )
