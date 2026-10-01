import numpy as np
import pytest
from rosclaw.growth.anchor_output_memory import AnchorOutputMemory

from rosclaw_soccer.rsi.kernel_guarded_step_execution import CompiledKernelStepMotor
from rosclaw_soccer.rsi.kernel_guarded_step_execution import make_preview as parent_preview
from rosclaw_soccer.rsi.output_memory_accelerated_learning import fit_update as accelerated
from rosclaw_soccer.rsi.output_memory_motor_learning import fit_update as historical
from rosclaw_soccer.rsi.output_memory_step_motor import (
    CompiledOutputMemoryMotor,
    encoder_identity,
    initial_model,
    make_preview,
)
from tests.rsi.test_kernel_guarded_step_motor import candidate  # noqa: F401
from tests.rsi.test_step_motor_network import model  # noqa: F401


def test_default_rate_exactly_reproduces_historical_numeric_policy(candidate):  # noqa: F811
    parent, anchors = candidate
    original = CompiledKernelStepMotor(parent_preview(parent))
    states = np.stack([np.concatenate((original.features(v)[:134], [1.0])) for v in anchors])
    memory = AnchorOutputMemory(
        states,
        np.stack([original.raw_mean(v, 1) for v in anchors]),
        bandwidth=1e-4,
        encoder_hash=encoder_identity(parent),
        parent_policy_hash=parent["model_hash"],
        evidence_hash="sha256:" + "a" * 64,
    )
    before = initial_model(parent, memory.to_dict())
    decoder = CompiledOutputMemoryMotor(make_preview(before))
    rng = np.random.default_rng(343)
    n = 960
    x = rng.normal(size=(n, 134))
    phase = np.tile(np.repeat(np.arange(3), 20), 16)
    mean = np.stack([decoder.raw_mean(v, int(p)) for v, p in zip(x, phase, strict=True)])
    noise = rng.normal(size=(n, 12))
    arrays = dict(
        observation=x,
        phase_index=phase,
        latent_action=mean + 0.1 * noise,
        old_log_probability=np.sum(-0.5 * noise**2 - np.log(0.1) - 0.5 * np.log(2 * np.pi), axis=1),
        terminal_return=np.repeat(np.where(np.arange(16) % 2, 3.0, -2.0), 60),
        std_raw=np.full(n, 0.1),
        trajectory_index=np.repeat(np.arange(16), 60),
    )
    a = historical(before, arrays, batch_hash="sha256:" + "b" * 64)
    b = accelerated(before, arrays, batch_hash="sha256:" + "b" * 64, learning_rate=1e-4)
    for key in ("residual_layers", "critic_readout", "frozen_parent", "output_memory"):
        assert a[key] == b[key]
    for key in ("full_batch_loss_history", "backtracking_reductions", "exact_mean_latent_kl"):
        assert a["learning_receipt"][key] == b["learning_receipt"][key]
    c = accelerated(before, arrays, batch_hash="sha256:" + "b" * 64, learning_rate=4e-4)
    assert c["learning_receipt"]["learning_rate"] == 4e-4
    assert 0 <= c["learning_receipt"]["exact_mean_latent_kl"] <= 0.005
    assert c["frozen_parent"] == before["frozen_parent"]
    assert c["output_memory"] == before["output_memory"]
    updated = CompiledOutputMemoryMotor(make_preview(c))
    assert all(np.array_equal(updated.raw_mean(v, 1), original.raw_mean(v, 1)) for v in anchors)
    with pytest.raises(ValueError, match="actual current"):
        accelerated(c, arrays, batch_hash="sha256:" + "c" * 64)
