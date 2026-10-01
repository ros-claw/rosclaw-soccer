import numpy as np
import pytest

from rosclaw_soccer.rsi.smooth_memory_learning import fit_update
from rosclaw_soccer.rsi.smooth_memory_motor import (
    CompiledSmoothMemoryMotor,
    make_preview,
    make_sampling_view,
)
from tests.rsi.test_kernel_guarded_step_motor import candidate  # noqa: F401
from tests.rsi.test_smooth_memory_motor import smooth_parent  # noqa: F401
from tests.rsi.test_step_motor_network import model  # noqa: F401


def test_all_layers_learn_using_conditional_not_white_likelihood(smooth_parent, candidate):  # noqa: F811
    rng = np.random.default_rng(348)
    n = 8 * 270
    x = rng.normal(size=(n, 134))
    phase = np.tile(np.repeat(np.arange(3), 90), 8)
    actions, probabilities = [], []
    for group in range(8):
        decoder = CompiledSmoothMemoryMotor(
            make_preview(make_sampling_view(smooth_parent, seed=348 + group, std=0.1, rho=0.9))
        )
        for frame in range(30, 300):
            index = group * 270 + frame - 30
            action, logp = decoder.latent_sample(x[index], frame, int(phase[index]))
            actions.append(action)
            probabilities.append(logp)
    arrays = dict(
        observation=x,
        phase_index=phase,
        latent_action=np.stack(actions),
        old_log_probability=np.asarray(probabilities),
        terminal_return=np.repeat(np.where(np.arange(8) % 2, 3.0, -2.0), 270),
        std_raw=np.full(n, 0.1),
        trajectory_index=np.repeat(np.arange(8), 270),
    )
    learned = fit_update(smooth_parent, arrays, batch_hash="sha256:" + "b" * 64)
    assert learned["generation"] == 1
    assert learned["frozen_parent"] == smooth_parent["frozen_parent"]
    assert all(
        a["weight"] != b["weight"]
        for a, b in zip(smooth_parent["residual_layers"], learned["residual_layers"], strict=True)
    )
    assert learned["learning_receipt"]["algorithm"] == "SMOOTH_MEMORY_AR1_PPO_MC_TERMINAL"
    for key in ("exact_mean_conditional_kl", "exact_mean_marginal_kl"):
        assert 0 <= learned["learning_receipt"][key] <= 0.005
    updated = CompiledSmoothMemoryMotor(make_preview(learned))
    old = CompiledSmoothMemoryMotor(make_preview(smooth_parent))
    _, anchors = candidate
    assert all(np.array_equal(updated.raw_mean(v, 1), old.raw_mean(v, 1)) for v in anchors)
    with pytest.raises(ValueError, match="history-conditioned"):
        fit_update(smooth_parent, arrays, batch_hash="sha256:" + "c" * 64, rho=0)
    with pytest.raises(ValueError, match="history-conditioned"):
        fit_update(learned, arrays, batch_hash="sha256:" + "d" * 64)
    incomplete = {k: v[:-1] for k, v in arrays.items()}
    with pytest.raises(ValueError, match="complete ordered"):
        fit_update(smooth_parent, incomplete, batch_hash="sha256:" + "e" * 64)
