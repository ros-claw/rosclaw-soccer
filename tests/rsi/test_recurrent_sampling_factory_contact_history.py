"""Causal contact-phase scans, not actual physical contact evidence."""

import numpy as np

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


def test_contact_before_during_and_after_first_touch_matches_original(learned):  # noqa: F811
    factory = RecurrentSamplingEpisodeFactory(learned[-1])
    policy = factory.preview(factory.sampling_view(seed=773))
    reference = CompiledRecurrentSamplingMotor(policy)
    actual = factory.bind(policy)
    untouched = factory.bind(policy)
    trace = long_body()
    previous = np.zeros(12)
    for frame in range(300):
        options = boundary(frame, previous)
        # Previous completed forces become observable at frame 60. The
        # three resulting phases must all affect the same causal context.
        if 60 <= frame < 65:
            options["previous_contact_forces"] = np.array([10.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        left = actual.delta_at_frame(policy, trace, **options)
        right = reference.delta_at_frame(policy, trace, **options)
        np.testing.assert_array_equal(left, right)
        np.testing.assert_array_equal(actual.hidden_state, reference.hidden_state)
        for field in TRACE_FIELDS:
            np.testing.assert_array_equal(
                actual.sampled_transition[field], reference.sampled_transition[field]
            )
        assert actual._memory.first_contact_frame == (None if frame < 60 else 59)
        assert actual._memory.last_frame == frame
        assert untouched._memory.first_contact_frame is None
        assert untouched._memory.last_frame == -1
        assert untouched._recurrent.next_index == 0
        previous = left
    fresh = factory.bind(policy)
    assert fresh._memory.first_contact_frame is None
    assert fresh._memory.last_frame == -1
    assert fresh._recurrent.next_index == 0
