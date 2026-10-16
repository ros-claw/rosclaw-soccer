"""Synthetic full-decoder parity; never physical skill evidence."""

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


def test_generated_seed_preview_keeps_original_decoder_and_fresh_episode_state(learned):  # noqa: F811
    factory = RecurrentSamplingEpisodeFactory(learned[-1])
    generated = factory.preview_from_seed(seed=773)
    assert generated == factory.preview(factory.sampling_view(seed=773))
    reference = CompiledRecurrentSamplingMotor(generated)
    actual = factory.bind(generated)
    untouched = factory.bind(factory.preview_from_seed(seed=773))
    trace, previous = long_body(), np.zeros(12)
    for frame in range(300):
        options = boundary(frame, previous)
        previous = actual.delta_at_frame(generated, trace, **options)
        np.testing.assert_array_equal(
            previous, reference.delta_at_frame(generated, trace, **options)
        )
        np.testing.assert_array_equal(actual.hidden_state, reference.hidden_state)
        for key in TRACE_FIELDS:
            np.testing.assert_array_equal(
                actual.sampled_transition[key], reference.sampled_transition[key]
            )
    np.testing.assert_array_equal(untouched.hidden_state, np.zeros(64))
    assert untouched._recurrent.next_index == 0
    assert untouched._memory is not actual._memory


def test_generated_document_tamper_still_rejected_by_original_bind(learned):  # noqa: F811
    factory = RecurrentSamplingEpisodeFactory(learned[-1])
    original = factory.preview_from_seed(seed=773)
    changed = copy.deepcopy(original)
    changed["step_motor_proof"]["model"]["mean_model"]["critic_parameters"]["bias_1"][0] += 0.01
    with pytest.raises(ValueError):
        factory.bind(changed)
    assert factory.preview_from_seed(seed=773) == original
    factory._prototype._policy_hash = "changed"
    with pytest.raises(ValueError, match="prototype"):
        factory.preview_from_seed(seed=773)
