"""Reference sampling parity and state isolation, not native physics evidence."""

import copy

import numpy as np
import pytest

from rosclaw_soccer.rsi.proposal_memory_learning import fit_update
from rosclaw_soccer.rsi.proposal_memory_motor import initial_model
from rosclaw_soccer.rsi.proposal_sampling_episode_factory import (
    ProposalSamplingEpisodeFactory,
    compilation_contract,
    validate_compilation_contract,
)
from rosclaw_soccer.rsi.proposal_sampling_motor import (
    CompiledProposalSamplingMotor,
    make_preview,
    make_sampling_view,
)
from tests.rsi.test_current_memory_motor import current  # noqa: F401
from tests.rsi.test_kernel_guarded_step_motor import candidate  # noqa: F401
from tests.rsi.test_phase_balanced_proposal_motor import imbalanced_complete_batch
from tests.rsi.test_smooth_memory_motor import smooth_parent  # noqa: F401
from tests.rsi.test_step_motor_network import model  # noqa: F401


@pytest.mark.parametrize("learned", [False, True])
def test_original_mean_and_noise_exact_with_private_episode_histories(
    current,  # noqa: F811
    smooth_parent,  # noqa: F811
    learned,
):
    mean = initial_model(current[0], maximum_mean_kl=0.05)
    if learned:
        mean = fit_update(
            mean,
            imbalanced_complete_batch(smooth_parent),
            batch_hash="sha256:" + "b" * 64,
            numeric_implementation="bounded_snapshot",
        )
    factory = ProposalSamplingEpisodeFactory(mean)
    policy = make_preview(make_sampling_view(mean, seed=17))
    original = CompiledProposalSamplingMotor(policy)
    left, right = factory.bind(policy), factory.bind(policy)
    provenance = compilation_contract(policy)
    validate_compilation_contract(provenance, policy)
    for key, value in (("hardware_authorized", True), ("actor_weights_changed", 0)):
        forged_contract = {**provenance, key: value}
        with pytest.raises(ValueError):
            validate_compilation_contract(forged_contract, policy)
    assert left._memory is not right._memory
    assert left._parent._memory is not right._parent._memory
    assert left._parent._warm is not right._parent._warm
    assert left._layers is not right._layers
    assert not left._noise.flags.writeable
    assert left._noise is not right._noise
    for frame in range(30, 300):
        x = np.full(134, frame / 1000)
        phase = 0 if frame < 59 else 1 if frame < 79 else 2
        a, density = original.latent_sample(x, frame, phase)
        for decoder in (left, right):
            b, actual_density = decoder.latent_sample(x, frame, phase)
            assert np.array_equal(a, b)
            assert density == actual_density
    for key, value in (("policy_hash", "wrong"), ("promotion_authorized", True)):
        forged = copy.deepcopy(policy)
        if key == "policy_hash":
            forged[key] = value
        else:
            forged["step_motor_proof"]["model"][key] = value
        with pytest.raises(ValueError):
            factory.bind(forged)
    other = initial_model(current[0], maximum_mean_kl=0.1)
    with pytest.raises(ValueError, match="same-mean"):
        factory.bind(make_preview(make_sampling_view(other, seed=17)))
