import copy

import numpy as np
import pytest

from rosclaw_soccer.rsi.compiled_step_inference import CompiledStepMotor
from rosclaw_soccer.rsi.kernel_guarded_step_execution import CompiledKernelStepMotor, make_preview
from rosclaw_soccer.rsi.kernel_guarded_step_execution import make_sampling_view as kernel_sampling
from rosclaw_soccer.rsi.kernel_guarded_step_network import (
    fit_update,
    initial_model,
    latents,
    validate_model,
    warm_means,
)
from rosclaw_soccer.rsi.step_motor_execution import make_preview as warm_preview
from rosclaw_soccer.rsi.stochastic_step_execution import make_preview as sampled_preview
from rosclaw_soccer.rsi.stochastic_step_execution import make_sampling_view as old_sampling
from rosclaw_soccer.sim.contracts import hash_json
from tests.rsi.test_step_motor_execution import body
from tests.rsi.test_step_motor_network import model  # noqa: F401


@pytest.fixture(scope="module")
def candidate(model):  # noqa: F811
    rng = np.random.default_rng(331)
    encoder = dict(
        base_model=model,
        frozen_random_features=dict(
            weight=(rng.normal(size=(249, 134)) / np.sqrt(134)).tolist(),
            bias=(rng.normal(size=249) * 0.1).tolist(),
        ),
    )
    anchors = rng.normal(size=(20, 134))
    return initial_model(encoder, anchors, anchor_bank_hash="sha256:" + "a" * 64), anchors


def test_all_success_states_remain_protected_without_removing_a_linear_span(candidate):
    value, anchors = candidate
    guard = validate_model(value)
    phi = latents(value, anchors)
    assert np.array_equal(guard.gates(phi[:, :134]), np.zeros(20))
    assert np.all(guard.gates(latents(value, anchors + 1)[:, :134]) > 0)
    decoder = CompiledKernelStepMotor(make_preview(value))
    assert np.allclose(decoder.features(anchors[0]), phi[0], atol=1e-12, rtol=0)
    assert decoder._guard.gate(decoder.features(anchors[0])[:134]) == 0


def test_zero_head_and_gaussian_view_reproduce_the_actual_parent_decoder(candidate):
    value, _ = candidate
    policy = make_preview(value)
    warm = warm_preview(value["encoder"]["base_model"])
    decoder = CompiledKernelStepMotor(policy)
    reference = CompiledStepMotor.from_legacy_preview(warm)
    previous = np.zeros(12)
    observation = body()
    for frame in range(34):
        kwargs = dict(
            frame=frame,
            nominal_target=np.zeros(29),
            baseline=np.zeros(12),
            limits=np.tile([-1.0, 1.0], (12, 1)),
            previous=previous,
            previous_contact_forces=np.zeros(6),
        )
        actual = decoder.delta_at_frame(policy, observation, **kwargs)
        assert np.array_equal(actual, reference.delta_at_frame(warm, observation, **kwargs))
        previous = actual
    sampled = CompiledKernelStepMotor(make_preview(kernel_sampling(value, seed=331, std=0.1)))
    old = CompiledStepMotor.from_legacy_preview(
        sampled_preview(old_sampling(value["encoder"]["base_model"], seed=331, std=0.1))
    )
    z, density = sampled.latent_sample(np.zeros(134), 30, 0)
    expected, old_density = old.latent_sample(np.zeros(134), 30)
    assert np.array_equal(z, expected) and density == old_density


def test_real_optimizer_changes_heads_and_rejects_stale_parent_data(candidate):
    value, _ = candidate
    rng = np.random.default_rng(332)
    n = 960
    x = rng.normal(size=(n, 134))
    means = warm_means(value, latents(value, x))
    noise = rng.normal(size=(n, 12))
    std = np.full(n, 0.1)
    groups = np.repeat(np.arange(16), 60)
    arrays = dict(
        observation=x,
        latent_action=means + 0.1 * noise,
        old_log_probability=np.sum(-0.5 * noise**2 - np.log(0.1) - 0.5 * np.log(2 * np.pi), axis=1),
        phase_index=np.tile(np.repeat(np.arange(3), 20), 16),
        std_raw=std,
        terminal_return=np.repeat(np.where(np.arange(16) % 2, 3.0, -2.0), 60),
        trajectory_index=groups,
    )
    updated = fit_update(value, arrays, batch_hash="sha256:" + "b" * 64)
    assert updated["generation"] == 1
    assert updated["parent_model_hash"] == value["model_hash"]
    assert updated["encoder"] == value["encoder"]
    assert updated["anchor_guard"] == value["anchor_guard"]
    assert np.any(np.asarray(updated["actor_readout"]) != 0)
    assert np.any(np.asarray(updated["critic_readout"]) != 0)
    assert updated["learning_receipt"]["exact_mean_latent_kl"] <= 0.005
    with pytest.raises(ValueError, match="actual current parent"):
        fit_update(updated, arrays, batch_hash="sha256:" + "c" * 64)


@pytest.mark.parametrize(
    "field", ["physics_qualified", "promotion_authorized", "hardware_authorized"]
)
def test_resealed_artifact_cannot_grant_authority(candidate, field):
    changed = copy.deepcopy(candidate[0])
    changed[field] = True
    changed["model_hash"] = hash_json({k: v for k, v in changed.items() if k != "model_hash"})
    with pytest.raises(ValueError):
        validate_model(changed)
