"""Synthetic optimizer tests, not physics outcomes or training achievements."""

import copy
import json

import numpy as np
import pytest

from rosclaw_soccer.rsi.cerebellar_actor_critic import (
    fit_update,
    initialize_checkpoint,
    validate_checkpoint,
)
from rosclaw_soccer.rsi.cerebellar_recovery_return import reward_contract
from rosclaw_soccer.rsi.cerebellar_torque_policy import initialize
from rosclaw_soccer.sim.contracts import hash_json

WORLD = "sha256:" + "a" * 64
BATCH = "sha256:" + "b" * 64


def mean(model, x):
    for i, layer in enumerate(model["layers"]):
        x = x @ np.asarray(layer["weight"]).T + np.asarray(layer["bias"])
        if i < len(model["layers"]) - 1:
            x = np.tanh(x)
    return x


def batch(model, *, seed=92, all_failed=False):
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 0.3, (6000, 127))
    x[:, -1] = np.tile((3000 - np.arange(3000)) / 3000, 2)
    mu = mean(model, x)
    z = mu + 0.05 * rng.normal(size=(6000, 29))
    return dict(
        behavior_model_hash=model["model_hash"],
        reward_contract_hash=reward_contract()["contract_hash"],
        features=x,
        latent_action=z,
        old_log_probability=np.sum(
            -0.5 * ((z - mu) / 0.05) ** 2 - np.log(0.05) - 0.5 * np.log(2 * np.pi), axis=1
        ),
        terminal_return=np.repeat([-10.0, -10.0 if all_failed else 2.0], 3000),
        trajectory_index=np.repeat(np.arange(2), 3000),
        terminated=np.ones(2, dtype=np.bool_),
        truncated=np.zeros(2, dtype=np.bool_),
    )


@pytest.fixture(scope="module")
def setup():
    actor = initialize(seed=42, world_hash=WORLD, time_aware=True)
    data = batch(actor)
    anchors = data["features"][:100].copy()
    checkpoint = initialize_checkpoint(actor, anchor_features=anchors)
    return checkpoint, data, anchors


def test_zero_step_checkpoint_is_json_only_and_has_no_authority(setup):
    checkpoint, _, _ = setup
    restored = json.loads(json.dumps(checkpoint))
    validate_checkpoint(restored)
    assert restored == checkpoint
    assert restored["optimizer"]["steps"] == 0
    for name in ("physics_qualified", "hardware_authorized", "runtime_execution_authorized"):
        assert not restored[name]


def test_actual_numeric_update_changes_actor_critic_and_preserves_old_objects(setup):
    checkpoint, data, anchors = setup
    original = copy.deepcopy(checkpoint)
    result = fit_update(checkpoint, data, batch_hash=BATCH, anchor_features=anchors, steps=2)
    receipt = result["learning_receipt"]
    assert receipt["accepted_joint_optimizer_steps"] == 2
    assert receipt["nonzero_actor_gradient_steps"] == 2
    assert result["actor"]["model_hash"] != checkpoint["actor"]["model_hash"]
    assert result["critic"] != checkpoint["critic"]
    assert result["anchor_actor"] == checkpoint["anchor_actor"]
    assert result["optimizer"]["steps"] == result["actor"]["optimizer_steps"] == 2
    assert checkpoint == original
    assert receipt["physical_provenance_verified_by_optimizer"] is False
    assert receipt["physical_qualification_claimed"] is False
    assert receipt["complete_numeric_episode_count"] == 2
    assert receipt["substep_sample_count"] == 6000
    assert receipt["kl"]["anchor_mean"] <= 0.001


def test_adam_and_critic_continue_through_json_and_new_behavior_batch(setup):
    checkpoint, data, anchors = setup
    first = fit_update(checkpoint, data, batch_hash=BATCH, anchor_features=anchors, steps=1)
    new_data = batch(first["actor"], seed=93)
    continued = fit_update(first, new_data, batch_hash=BATCH, anchor_features=anchors, steps=1)
    restored = fit_update(
        json.loads(json.dumps(first)), new_data, batch_hash=BATCH, anchor_features=anchors, steps=1
    )
    assert continued == restored
    assert continued["optimizer"]["steps"] == 2
    assert continued["generation"] == 2
    assert any(np.any(np.asarray(slot["exp_avg"]) != 0) for slot in first["optimizer"]["slots"])
    assert continued["critic"] != first["critic"]
    # Deliberately reset Adam in a synthetic counterexample: same weights are not
    # enough for continuity. This is NEVER a supported physical checkpoint path.
    reset = copy.deepcopy(first)
    for slot in reset["optimizer"]["slots"]:
        for key in slot:
            slot[key] = np.zeros_like(slot[key]).tolist()
    reset["checkpoint_hash"] = hash_json({k: v for k, v in reset.items() if k != "checkpoint_hash"})
    reset_result = fit_update(reset, new_data, batch_hash=BATCH, anchor_features=anchors, steps=1)
    assert continued["actor"]["model_hash"] != reset_result["actor"]["model_hash"]


def test_equal_failures_do_not_get_centered_into_zero_actor_advantage(setup):
    checkpoint, _, anchors = setup
    result = fit_update(
        checkpoint,
        batch(checkpoint["actor"], all_failed=True),
        batch_hash=BATCH,
        anchor_features=anchors,
        steps=1,
    )
    assert result["learning_receipt"]["nonzero_actor_gradient_steps"] == 1
    assert result["actor"]["model_hash"] != checkpoint["actor"]["model_hash"]


@pytest.mark.parametrize(
    "kind",
    ["old_density", "behavior", "clock", "incomplete", "returns", "bootstrap", "order", "float32"],
)
def test_invalid_or_off_policy_numeric_batches_rejected_before_optimization(setup, kind):
    checkpoint, original, anchors = setup
    data = copy.deepcopy(original)
    if kind == "old_density":
        data["old_log_probability"][0] += 0.01
    elif kind == "behavior":
        data["behavior_model_hash"] = BATCH
    elif kind == "clock":
        data["features"][10, -1] = 1
    elif kind == "incomplete":
        data["features"] = data["features"][:-1]
    elif kind == "returns":
        data["terminal_return"][10] = 0
    elif kind == "bootstrap":
        data["truncated"][0] = True
    elif kind == "float32":
        data["latent_action"] = data["latent_action"].astype(np.float32)
    else:
        data["trajectory_index"] = data["trajectory_index"][::-1]
    with pytest.raises(ValueError):
        fit_update(checkpoint, data, batch_hash=BATCH, anchor_features=anchors, steps=1)


def test_anchor_change_or_silent_optimizer_reset_is_not_allowed(setup):
    checkpoint, data, anchors = setup
    changed = anchors.copy()
    changed[0, 0] += 0.01
    with pytest.raises(ValueError, match="anchor"):
        fit_update(checkpoint, data, batch_hash=BATCH, anchor_features=changed)
    learned = copy.deepcopy(checkpoint["actor"])
    learned["optimizer_steps"] = 1
    learned["model_hash"] = hash_json({k: v for k, v in learned.items() if k != "model_hash"})
    with pytest.raises(ValueError, match="reset"):
        initialize_checkpoint(learned, anchor_features=anchors)


def test_kl_rejection_restores_both_weights_and_moments(setup, monkeypatch):
    import torch

    checkpoint, data, anchors = setup
    original_step = torch.optim.Adam.step

    def explode(optimizer, *args, **kwargs):
        original_step(optimizer, *args, **kwargs)
        with torch.no_grad():
            optimizer.param_groups[0]["params"][0].add_(50)
            optimizer.param_groups[0]["params"][5].add_(2)

    monkeypatch.setattr(torch.optim.Adam, "step", explode)
    result = fit_update(checkpoint, data, batch_hash=BATCH, anchor_features=anchors, steps=4)
    assert result["learning_receipt"]["attempted_optimizer_steps"] == 1
    assert result["learning_receipt"]["accepted_joint_optimizer_steps"] == 0
    assert result["learning_receipt"]["rejected_trial_rolled_back_with_adam_state"] is True
    assert result["actor"] == checkpoint["actor"]
    assert result["critic"] == checkpoint["critic"]
    assert result["optimizer"] == checkpoint["optimizer"]
    with pytest.raises(ValueError, match="already consumed"):
        fit_update(result, data, batch_hash=BATCH, anchor_features=anchors, steps=1)
