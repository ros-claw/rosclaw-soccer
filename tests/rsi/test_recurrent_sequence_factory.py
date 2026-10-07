"""Synthetic extraction parity is not physics or Foundation qualification."""

import copy

import numpy as np
import pytest

from rosclaw_soccer.rsi.recurrent_sample_sequence import extract_sequence
from rosclaw_soccer.rsi.recurrent_sampling_episode_factory import RecurrentSamplingEpisodeFactory
from rosclaw_soccer.rsi.recurrent_sampling_motor import TRACE_FIELDS
from rosclaw_soccer.rsi.recurrent_success_motor import STATE_FIELD
from tests.rsi.test_current_memory_motor import current  # noqa: F401
from tests.rsi.test_imitation_proposal_motor import imitation_parent  # noqa: F401
from tests.rsi.test_kernel_guarded_step_motor import candidate  # noqa: F401
from tests.rsi.test_recurrent_clipped_motor import learned  # noqa: F401
from tests.rsi.test_recurrent_sample_sequence import sample  # noqa: F401
from tests.rsi.test_smooth_memory_motor import smooth_parent  # noqa: F401
from tests.rsi.test_step_motor_network import model  # noqa: F401


def test_every_learning_array_matches_original_and_labels_stay_out_of_actor_input(sample):  # noqa: F811
    view, trace, limits = sample
    before = copy.deepcopy(trace)
    factory = RecurrentSamplingEpisodeFactory(view["mean_model"])
    reference = extract_sequence(view, trace, motor_limits=limits, terminal_mc_return=-123.0)
    for reward in (-123.0, 12.0):
        actual = extract_sequence(
            view, trace, motor_limits=limits, terminal_mc_return=reward, episode_factory=factory
        )
        assert set(actual) == set(reference)
        for key in reference.keys() - {"returns"}:
            assert actual[key].dtype == reference[key].dtype
            np.testing.assert_array_equal(actual[key], reference[key])
        assert np.all(actual["returns"] == reward)
        assert set(actual["context"][:, -1]) == {0, 1, 2}
        actual["context"][:] = 123
        actual["latent_actions"][:] = 123
    for key in trace:
        np.testing.assert_array_equal(trace[key], before[key])
    np.testing.assert_array_equal(factory._prototype.hidden_state, np.zeros(64))


def test_factory_still_rejects_each_changed_actual_boundary_and_complete_critic(sample):  # noqa: F811
    view, trace, limits = sample
    factory = RecurrentSamplingEpisodeFactory(view["mean_model"])
    for key in (STATE_FIELD, "motor_delta_rad", *TRACE_FIELDS[:3]):
        bad = copy.deepcopy(trace)
        bad[key][35, 0, 0] += 0.001
        with pytest.raises(ValueError, match="reconstruct"):
            extract_sequence(
                view, bad, motor_limits=limits, terminal_mc_return=-123.0, episode_factory=factory
            )
    changed = copy.deepcopy(view)
    changed["mean_model"]["critic_parameters"]["bias_1"][0] += 1
    with pytest.raises(ValueError, match="actor/critic/parent"):
        extract_sequence(
            changed, trace, motor_limits=limits, terminal_mc_return=-123.0, episode_factory=factory
        )


@pytest.mark.parametrize("factory", [object(), lambda _: None])
def test_no_external_factory_callback(factory):
    with pytest.raises(ValueError, match="exact private"):
        extract_sequence({}, {}, motor_limits=None, terminal_mc_return=0.0, episode_factory=factory)
