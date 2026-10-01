import copy

import numpy as np
import pytest
from rosclaw.growth.anchor_output_memory import AnchorOutputMemory
from rosclaw.growth.correlated_exploration import conditional_mean, conditional_scale

from rosclaw_soccer.rsi.kernel_guarded_step_execution import CompiledKernelStepMotor
from rosclaw_soccer.rsi.kernel_guarded_step_execution import make_preview as kernel_preview
from rosclaw_soccer.rsi.output_memory_step_motor import encoder_identity
from rosclaw_soccer.rsi.output_memory_step_motor import initial_model as memory_initial
from rosclaw_soccer.rsi.smooth_memory_motor import (
    CompiledSmoothMemoryMotor,
    initial_model,
    make_preview,
    make_sampling_view,
    mean_model,
)
from rosclaw_soccer.sim.contracts import hash_json
from tests.rsi.test_kernel_guarded_step_motor import candidate  # noqa: F401
from tests.rsi.test_step_motor_network import model  # noqa: F401


@pytest.fixture
def smooth_parent(candidate):  # noqa: F811
    parent, anchors = candidate
    original = CompiledKernelStepMotor(kernel_preview(parent))
    memory = AnchorOutputMemory(
        np.stack([np.concatenate((original.features(v)[:134], [1.0])) for v in anchors]),
        np.stack([original.raw_mean(v, 1) for v in anchors]),
        bandwidth=1e-4,
        encoder_hash=encoder_identity(parent),
        parent_policy_hash=parent["model_hash"],
        evidence_hash="sha256:" + "a" * 64,
    )
    return initial_model(memory_initial(parent, memory.to_dict()))


def test_stationary_sampling_has_the_actual_conditional_density(smooth_parent):
    view = make_sampling_view(smooth_parent, seed=42, std=0.1, rho=0.9)
    decoder = CompiledSmoothMemoryMotor(make_preview(view))
    rng = np.random.default_rng(348)
    previous_mean = previous_action = None
    for frame in range(30, 80):
        x = rng.normal(size=134)
        mean = decoder.raw_mean(x, 1)
        action, logp = decoder.latent_sample(x, frame, 1)
        conditional = (
            mean if frame == 30 else conditional_mean(mean, previous_mean, previous_action, 0.9)
        )
        std = conditional_scale(0.1, 0.9, first=frame == 30)
        expected = float(
            np.sum(
                -0.5 * ((action - conditional) / std) ** 2 - np.log(std) - 0.5 * np.log(2 * np.pi)
            )
        )
        assert logp == pytest.approx(expected, abs=1e-10)
        previous_mean, previous_action = mean, action
    with pytest.raises(ValueError, match="frame"):
        decoder.latent_sample(x, 300, 1)


def test_white_limit_preserves_all_original_seedsequence_actions(smooth_parent):
    decoder = CompiledSmoothMemoryMotor(
        make_preview(make_sampling_view(smooth_parent, seed=4, std=0.1, rho=0))
    )
    x = np.full(134, 0.25)
    for frame in (30, 31, 99, 299):
        noise = np.random.default_rng(np.random.SeedSequence(4, spawn_key=(frame,))).normal(size=12)
        action, _ = decoder.latent_sample(x, frame, 1)
        assert np.array_equal(action, decoder.raw_mean(x, 1) + 0.1 * noise)


def test_resealed_motion_authority_or_changed_correlation_is_rejected(smooth_parent):
    view = make_sampling_view(smooth_parent, seed=4, std=0.1)
    for key, value in (("hardware_authorized", True), ("rho", 1), ("seed", True)):
        forged = copy.deepcopy(view)
        forged[key] = value
        forged.pop("model_hash")
        forged["model_hash"] = hash_json(forged)
        with pytest.raises(ValueError):
            mean_model(forged)


def test_compiled_exploration_does_not_alias_the_callers_commitment(smooth_parent):
    view = make_sampling_view(smooth_parent, seed=4, std=0.1)
    policy = make_preview(view)
    decoder = CompiledSmoothMemoryMotor(policy)
    x = np.zeros(134)
    before = decoder.latent_sample(x, 31, 1)
    view["std_raw"] = 0.15
    view["rho"] = 0
    policy["step_motor_proof"]["model"]["seed"] = 999
    after = decoder.latent_sample(x, 31, 1)
    assert np.array_equal(before[0], after[0])
    assert before[1] == after[1]
