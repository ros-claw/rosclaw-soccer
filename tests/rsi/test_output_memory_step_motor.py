import copy

import numpy as np
import pytest
from rosclaw.growth.anchor_output_memory import AnchorOutputMemory

from rosclaw_soccer.rsi.kernel_guarded_step_execution import CompiledKernelStepMotor
from rosclaw_soccer.rsi.kernel_guarded_step_execution import make_preview as parent_preview
from rosclaw_soccer.rsi.output_memory_step_motor import (
    CompiledOutputMemoryMotor,
    encoder_identity,
    initial_model,
    make_preview,
    make_sampling_view,
    validate_model,
)
from rosclaw_soccer.sim.contracts import hash_json
from tests.rsi.test_kernel_guarded_step_motor import candidate  # noqa: F401
from tests.rsi.test_step_motor_network import model  # noqa: F401


def test_memory_preserves_later_parent_not_first_warm_start(candidate):  # noqa: F811
    zero, _ = candidate
    parent = copy.deepcopy(zero)
    parent["generation"] = 1
    parent["actor_readout"] = np.full((3, 12, 512), 0.01).tolist()
    parent["learning_receipt"] = dict(
        algorithm="KERNEL_GUARDED_PPO_CLIP_MC_TERMINAL",
        physical_batch_hash="sha256:" + "a" * 64,
        frozen_encoder=True,
        distributional_retention_guaranteed=False,
        exact_mean_latent_kl=0.0,
        promotion_authorized=False,
        hardware_authorized=False,
    )
    parent.pop("model_hash")
    parent["model_hash"] = hash_json(parent)
    decoder = CompiledKernelStepMotor(parent_preview(parent))
    old = CompiledKernelStepMotor(parent_preview(zero))
    observation = np.full(134, 3.0)
    reference = decoder.raw_mean(observation, 1)
    assert not np.array_equal(reference, old.raw_mean(observation, 1))
    context = np.concatenate((decoder.features(observation)[:134], [1.0]))
    memory = AnchorOutputMemory(
        context[None],
        reference[None],
        bandwidth=1e-4,
        encoder_hash=encoder_identity(parent),
        parent_policy_hash=parent["model_hash"],
        evidence_hash="sha256:" + "b" * 64,
    )
    bootstrap = initial_model(parent, memory.to_dict())
    assert np.array_equal(
        CompiledOutputMemoryMotor(make_preview(bootstrap)).raw_mean(observation, 1), reference
    )
    learned = copy.deepcopy(bootstrap)
    learned["generation"] = 1
    learned["residual_layers"][-1]["bias"] = np.ones(12).tolist()
    learned["learning_receipt"] = dict(
        algorithm="OUTPUT_MEMORY_MLP_PPO_MC_TERMINAL",
        physical_batch_hash="sha256:" + "c" * 64,
        learner_parent_hash=bootstrap["model_hash"],
        optimizer_source_hash="sha256:" + "d" * 64,
        completed_optimizer_steps=1,
        all_residual_layers_trainable=True,
        frozen_parent=True,
        distributional_retention_guaranteed=False,
        exact_mean_latent_kl=0.0,
        promotion_authorized=False,
        hardware_authorized=False,
    )
    learned.pop("model_hash")
    learned["model_hash"] = hash_json(learned)
    updated = CompiledOutputMemoryMotor(make_preview(learned))
    assert np.array_equal(updated.raw_mean(observation, 1), reference)
    assert not np.array_equal(
        updated.raw_mean(observation + 1, 1), decoder.raw_mean(observation + 1, 1)
    )
    with pytest.raises(ValueError, match="phase"):
        updated.raw_mean(observation, True)
    sampling = make_sampling_view(bootstrap, seed=342, std=0.1)
    stochastic = CompiledOutputMemoryMotor(make_preview(sampling))
    value, density = stochastic.latent_sample(observation, 30, 1)
    noise = np.random.default_rng(np.random.SeedSequence(342, spawn_key=(30,))).normal(size=12)
    assert np.array_equal(value, reference + 0.1 * noise)
    assert density == float(np.sum(-0.5 * noise**2 - np.log(0.1) - 0.5 * np.log(2 * np.pi)))
    forged = copy.deepcopy(bootstrap)
    forged["hardware_authorized"] = True
    forged.pop("model_hash")
    forged["model_hash"] = hash_json(forged)
    with pytest.raises(ValueError, match="SIM-only"):
        validate_model(forged)
