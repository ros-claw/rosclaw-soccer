"""Synthetic complete-model parity, not physics or real-model speed evidence."""

import copy

import numpy as np
import pytest

from rosclaw_soccer.rsi.recurrent_sampling_episode_factory import RecurrentSamplingEpisodeFactory
from rosclaw_soccer.rsi.recurrent_sampling_motor import TRACE_FIELDS, CompiledRecurrentSamplingMotor
from tests.rsi.test_current_memory_motor import current  # noqa: F401
from tests.rsi.test_imitation_proposal_motor import imitation_parent  # noqa: F401
from tests.rsi.test_kernel_guarded_step_motor import candidate  # noqa: F401
from tests.rsi.test_recurrent_clipped_motor import learned  # noqa: F401
from tests.rsi.test_recurrent_sampling_motor import long_body
from tests.rsi.test_recurrent_success_motor import boundary
from tests.rsi.test_smooth_memory_motor import smooth_parent  # noqa: F401
from tests.rsi.test_step_motor_network import model  # noqa: F401


@pytest.mark.parametrize("index", [0, -1])
def test_generated_seed_bind_matches_original_full_constructor_and_owns_state(learned, index):  # noqa: F811
    factory = RecurrentSamplingEpisodeFactory(learned[index])
    policy, actual = factory.generate_and_bind_seed(seed=773)
    reference = CompiledRecurrentSamplingMotor(policy)
    original = factory.preview(factory.sampling_view(seed=773))
    assert original == policy
    other_policy, untouched = factory.generate_and_bind_seed(seed=773)
    assert other_policy == policy and other_policy is not policy
    previous = np.zeros(12)
    trace = long_body()
    for frame in range(300):
        options = boundary(frame, previous)
        left = actual.delta_at_frame(policy, trace, **options)
        right = reference.delta_at_frame(policy, trace, **options)
        np.testing.assert_array_equal(left, right)
        np.testing.assert_array_equal(actual.hidden_state, reference.hidden_state)
        for field in TRACE_FIELDS:
            np.testing.assert_array_equal(
                actual.sampled_transition[field], reference.sampled_transition[field]
            )
        previous = left
    np.testing.assert_array_equal(untouched.hidden_state, np.zeros(64))
    assert untouched._recurrent.next_index == 0
    assert untouched._memory is not actual._memory
    assert untouched._behavior._memory is not actual._behavior._memory
    assert not actual._noise.flags.writeable
    factory._stable()
    # An owned generated document is not a privileged external bind token.
    altered = copy.deepcopy(other_policy)
    mean = altered["step_motor_proof"]["model"]["mean_model"]
    if "critic_parameters" in mean:
        mean["critic_parameters"]["bias_1"][0] += 0.01
    else:
        mean["parameters"]["head_bias"][0] += 0.01
    with pytest.raises(ValueError):
        factory.bind(altered)


def test_generated_seed_does_not_accept_external_models_or_bad_seeds(learned):  # noqa: F811
    factory = RecurrentSamplingEpisodeFactory(learned[-1])
    for seed in (True, -1, 2**32, 0.0, "773", None):
        with pytest.raises(ValueError, match="integer"):
            factory.generate_and_bind_seed(seed=seed)
    with pytest.raises(TypeError):
        factory.generate_and_bind_seed(seed=773, policy={})
    factory._prototype._last_draw[0] = 1
    with pytest.raises(ValueError, match="prototype"):
        factory.generate_and_bind_seed(seed=773)
