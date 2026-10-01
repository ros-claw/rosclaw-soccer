import numpy as np
import pytest

from rosclaw_soccer.rsi.kernel_full_batch_learning import fit_update
from rosclaw_soccer.rsi.kernel_guarded_step_network import latents, validate_model, warm_means
from tests.rsi.test_kernel_guarded_step_motor import candidate  # noqa: F401
from tests.rsi.test_step_motor_network import model  # noqa: F401


def test_full_batch_updates_are_auditable_and_keep_frozen_success_memory(candidate):  # noqa: F811
    parent, anchors = candidate
    rng = np.random.default_rng(335)
    n = 960
    x = rng.normal(size=(n, 134))
    means = warm_means(parent, latents(parent, x))
    noise = rng.normal(size=(n, 12))
    arrays = dict(
        observation=x,
        latent_action=means + 0.1 * noise,
        old_log_probability=np.sum(-0.5 * noise**2 - np.log(0.1) - 0.5 * np.log(2 * np.pi), axis=1),
        phase_index=np.tile(np.repeat(np.arange(3), 20), 16),
        std_raw=np.full(n, 0.1),
        terminal_return=np.repeat(np.where(np.arange(16) % 2, 3.0, -2.0), 60),
        trajectory_index=np.repeat(np.arange(16), 60),
    )
    updated = fit_update(parent, arrays, batch_hash="sha256:" + "b" * 64)
    receipt = updated["learning_receipt"]
    assert updated["generation"] == 1
    assert updated["encoder"] == parent["encoder"]
    assert updated["anchor_guard"] == parent["anchor_guard"]
    assert 0 < receipt["completed_optimizer_steps"] <= 40
    assert 0 <= receipt["exact_mean_latent_kl"] <= 0.005
    assert all(
        b < a
        for a, b in zip(
            receipt["full_batch_loss_history"], receipt["full_batch_loss_history"][1:], strict=False
        )
    )
    guard = validate_model(updated)
    assert np.array_equal(guard.gates(latents(updated, anchors)[:, :134]), np.zeros(len(anchors)))
    assert not updated["promotion_authorized"]
    with pytest.raises(ValueError, match="actual current parent"):
        fit_update(updated, arrays, batch_hash="sha256:" + "c" * 64)
