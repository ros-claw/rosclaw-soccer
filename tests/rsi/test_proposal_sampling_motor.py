"""Latent sampling law and commitments, not physical skill evidence."""

import copy

import numpy as np
import pytest

from rosclaw_soccer.rsi.proposal_memory_motor import initial_model
from rosclaw_soccer.rsi.proposal_sampling_motor import (
    CompiledProposalSamplingMotor,
    make_preview,
    make_sampling_view,
)
from rosclaw_soccer.sim.contracts import hash_json
from tests.rsi.test_current_memory_motor import current  # noqa: F401
from tests.rsi.test_kernel_guarded_step_motor import candidate  # noqa: F401
from tests.rsi.test_smooth_memory_motor import smooth_parent  # noqa: F401
from tests.rsi.test_step_motor_network import model  # noqa: F401


def test_sampling_matches_independent_stationary_noise_and_conditional_density(current):  # noqa: F811
    mean = initial_model(current[0], maximum_mean_kl=0.05)
    view = make_sampling_view(mean, seed=513)
    policy = make_preview(view)
    decoder = CompiledProposalSamplingMotor(policy)
    assert decoder._policy_hash == policy["policy_hash"]
    assert (
        policy["proposal_sampling_motor_proof"]["actual_behavior_model_hash"] == mean["model_hash"]
    )
    previous_noise = np.zeros(12)
    for frame in range(30, 300):
        x = np.full(134, frame / 1000)
        phase = 0 if frame < 59 else 1 if frame < 79 else 2
        noise = np.random.default_rng(np.random.SeedSequence(513, spawn_key=(frame,))).normal(
            size=12
        )
        scale = 0.1 if frame == 30 else 0.1 * np.sqrt(1 - 0.9**2)
        offset = np.zeros(12) if frame == 30 else 0.9 * previous_noise
        latent, logp = decoder.latent_sample(x, frame, phase)
        expected = decoder.raw_mean(x, phase) + offset + scale * noise
        np.testing.assert_allclose(latent, expected, atol=1e-14, rtol=0)
        expected_logp = np.sum(-0.5 * noise**2 - np.log(scale) - 0.5 * np.log(2 * np.pi))
        assert logp == pytest.approx(expected_logp, abs=1e-12)
        previous_noise = offset + scale * noise
    assert not decoder._noise.flags.writeable
    for frame in (True, 29, 300):
        with pytest.raises(ValueError):
            decoder.latent_sample(np.zeros(134), frame, 0)
    for key, value in (
        ("std_raw", 0.15),
        ("rho", 0.8),
        ("seed", True),
        ("source_hash", "sha256:" + "f" * 64),
        ("hardware_authorized", True),
        ("activation_ceiling", "REAL"),
    ):
        forged = copy.deepcopy(view)
        forged[key] = value
        forged.pop("model_hash")
        forged["model_hash"] = hash_json(forged)
        with pytest.raises(ValueError):
            make_preview(forged)
