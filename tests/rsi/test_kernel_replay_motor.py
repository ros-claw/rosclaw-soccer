import copy

import numpy as np
import pytest

from rosclaw_soccer.rsi.kernel_guarded_step_execution import CompiledKernelStepMotor
from rosclaw_soccer.rsi.kernel_guarded_step_execution import make_preview as parent_preview
from rosclaw_soccer.rsi.kernel_guarded_step_network import latents, warm_means
from rosclaw_soccer.rsi.kernel_replay_motor import (
    CompiledReplayStepMotor,
    fit_update,
    make_preview,
    validate_model,
)
from rosclaw_soccer.sim.contracts import hash_json
from tests.rsi.test_kernel_guarded_step_motor import candidate  # noqa: F401
from tests.rsi.test_step_motor_network import model  # noqa: F401


def test_replay_learns_with_separate_receipt_and_exact_old_memory(candidate):  # noqa: F811
    parent, anchors = candidate
    rng = np.random.default_rng(340)
    n = 960
    x = rng.normal(size=(n, 134))
    base = warm_means(parent, latents(parent, x))
    noise = rng.normal(size=(n, 12))
    arrays = dict(
        observation=x,
        phase_index=np.tile(np.repeat(np.arange(3), 20), 16),
        latent_action=base + 0.1 * noise,
        old_log_probability=np.sum(-0.5 * noise**2 - np.log(0.1) - 0.5 * np.log(2 * np.pi), axis=1),
        terminal_return=np.repeat(np.where(np.arange(16) % 2, 3.0, -2.0), 60),
        std_raw=np.full(n, 0.1),
        trajectory_index=np.repeat(np.arange(16), 60),
    )
    before = copy.deepcopy(parent)
    learned = fit_update(parent, arrays, batch_hash="sha256:" + "b" * 64)
    assert parent == before and learned["frozen_parent"] == before
    r = learned["learning_receipt"]
    assert r["algorithm"] == "AWR_INSPIRED_TERMINAL_GAUSSIAN_REGRESSION"
    assert 0 < r["completed_optimizer_steps"] <= 160
    assert 0 <= r["exact_mean_latent_kl"] <= 0.005
    history = r["full_batch_loss_history"]
    assert all(b < a for a, b in zip(history, history[1:], strict=False))
    decoder = CompiledReplayStepMotor(make_preview(learned))
    old = CompiledKernelStepMotor(parent_preview(parent))
    for observation in anchors:
        for phase in range(3):
            assert np.array_equal(
                decoder.raw_mean(observation, phase), old.raw_mean(observation, phase)
            )
    assert any(not np.array_equal(decoder.raw_mean(v, 1), old.raw_mean(v, 1)) for v in x[:5])
    for flag in ("hardware_authorized", "promotion_authorized"):
        forged = copy.deepcopy(learned)
        forged[flag] = True
        forged.pop("model_hash")
        forged["model_hash"] = hash_json(forged)
        with pytest.raises(ValueError, match="SIM-only"):
            validate_model(forged)
    stale = dict(arrays, old_log_probability=arrays["old_log_probability"] + 0.01)
    with pytest.raises(ValueError, match="warm behavior"):
        fit_update(parent, stale, batch_hash="sha256:" + "c" * 64)
