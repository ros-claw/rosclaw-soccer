"""Numeric migration proof, not physics or candidate qualification."""

import numpy as np
from rosclaw.growth.correlated_residual_gradient import (
    fit_correlated_residual,
    terminal_crossfit_advantages,
)

from rosclaw_soccer.rsi.smooth_memory_learning import fit_update
from rosclaw_soccer.rsi.smooth_memory_motor import (
    CompiledSmoothMemoryMotor,
    make_preview,
    make_sampling_view,
)
from tests.rsi.test_kernel_guarded_step_motor import candidate  # noqa: F401
from tests.rsi.test_smooth_memory_motor import smooth_parent  # noqa: F401
from tests.rsi.test_step_motor_network import model  # noqa: F401


def conditional_fixture_batch(parent):
    rng = np.random.default_rng(348)
    n = 8 * 270
    x = rng.normal(size=(n, 134))
    phase = np.tile(np.repeat(np.arange(3), 90), 8)
    actions, probabilities = [], []
    for g in range(8):
        sampled = CompiledSmoothMemoryMotor(
            make_preview(make_sampling_view(parent, seed=348 + g, std=0.1, rho=0.9))
        )
        for f in range(30, 300):
            i = g * 270 + f - 30
            action, probability = sampled.latent_sample(x[i], f, int(phase[i]))
            actions.append(action)
            probabilities.append(probability)
    arrays = dict(
        observation=x,
        phase_index=phase,
        latent_action=np.stack(actions),
        old_log_probability=np.asarray(probabilities),
        terminal_return=np.repeat(np.where(np.arange(8) % 2, 3.0, -2.0), 270),
        std_raw=np.full(n, 0.1),
        trajectory_index=np.repeat(np.arange(8), 270),
    )
    return arrays


def test_core_engine_matches_frozen_legacy_ar_update(smooth_parent):  # noqa: F811
    arrays = conditional_fixture_batch(smooth_parent)
    x, phase = arrays["observation"], arrays["phase_index"]
    n = len(x)
    old = fit_update(smooth_parent, arrays, batch_hash="sha256:" + "b" * 64)
    decoder = CompiledSmoothMemoryMotor(make_preview(smooth_parent))
    mean = np.stack([decoder.raw_mean(v, int(p)) for v, p in zip(x, phase, strict=True)])
    phi = np.stack([decoder.features(v) for v in x])
    context = np.column_stack((phi[:, :134], phase))
    gates = decoder._guard.gates(context)
    layers = [
        (np.asarray(v["weight"]), np.asarray(v["bias"])) for v in smooth_parent["residual_layers"]
    ]
    hidden = context
    for w, b in layers:
        hidden = np.tanh(hidden @ w.T + b)
    prepared = terminal_crossfit_advantages(
        phi, phase, arrays["trajectory_index"], arrays["terminal_return"]
    )
    new = fit_correlated_residual(
        layers=layers,
        context=context,
        baseline=mean - 0.05 * gates[:, None] * hidden,
        gates=gates,
        actions=arrays["latent_action"],
        marginal_std=arrays["std_raw"],
        first=np.arange(n) % 270 == 0,
        advantages=prepared["advantages"],
        old_log_probability=arrays["old_log_probability"],
    )
    # Exact equality is deliberately stricter than "same loss direction".
    assert new["layers"] == old["residual_layers"]
    receipt = old["learning_receipt"]
    assert new["full_batch_loss_history"] == receipt["full_batch_loss_history"]
    assert new["completed_optimizer_steps"] == receipt["completed_optimizer_steps"]
    assert prepared["critic_readout"].tolist() == old["critic_readout"]
    for key in ("exact_mean_conditional_kl", "exact_mean_marginal_kl"):
        # Compiled inference is per-row matrix-vector, rather than batch GEMM.
        assert np.isclose(new[key], receipt[key], atol=1e-12, rtol=0)
    assert new["physical_batch_verified"] is False
