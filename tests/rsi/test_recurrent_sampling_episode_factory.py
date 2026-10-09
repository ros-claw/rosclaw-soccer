"""Numeric synthetic parity is not actual MuJoCo or Foundation evidence."""

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


def test_complete_scan_matches_reference_and_new_episodes_stay_independent(learned):  # noqa: F811
    for mean in (learned[0], learned[-1]):
        factory = RecurrentSamplingEpisodeFactory(mean)
        for seed in (773, 798):
            policy = factory.preview(factory.sampling_view(seed=seed))
            reference = CompiledRecurrentSamplingMotor(policy)
            actual = factory.bind(policy)
            untouched = factory.bind(policy)
            previous = np.zeros(12)
            trace = long_body()
            for frame in range(300):
                options = boundary(frame, previous)
                left = actual.delta_at_frame(policy, trace, **options)
                right = reference.delta_at_frame(policy, trace, **options)
                np.testing.assert_array_equal(left, right)
                np.testing.assert_array_equal(actual.hidden_state, reference.hidden_state)
                assert actual._recurrent.next_index == reference._recurrent.next_index
                for field in TRACE_FIELDS:
                    np.testing.assert_array_equal(
                        actual.sampled_transition[field], reference.sampled_transition[field]
                    )
                previous = left
            np.testing.assert_array_equal(untouched.hidden_state, np.zeros(64))
            assert untouched._recurrent.next_index == 0
            assert not untouched.sampled_transition[TRACE_FIELDS[-1]][0]
            assert untouched._memory is not actual._memory
            assert untouched._behavior._memory is not actual._behavior._memory
            assert untouched._parent._memory is not actual._parent._memory
            assert untouched._parent._warm is not actual._parent._warm
            assert not actual._noise.flags.writeable
            after = factory.bind(policy)
            np.testing.assert_array_equal(after.hidden_state, np.zeros(64))
            assert after._recurrent.next_index == 0
            for frame in range(32):
                options = boundary(frame, np.zeros(12))
                np.testing.assert_array_equal(
                    after.delta_at_frame(policy, trace, **options),
                    untouched.delta_at_frame(policy, trace, **options),
                )
        contract = factory.contract()
        assert contract["actor_or_critic_weights_changed"] is False
        assert contract["native_transport_qualification_performed"] is False
        contract["source_pins"].clear()
        assert factory.contract()["source_pins"]


def test_changed_preview_and_fixed_mean_rejected_before_binding(learned):  # noqa: F811
    factory = RecurrentSamplingEpisodeFactory(learned[-1])
    policy = factory.preview(factory.sampling_view(seed=773))
    changed = copy.deepcopy(policy)
    changed["step_motor_proof"]["model"]["mean_model"]["critic_parameters"]["bias_1"][0] += 0.01
    with pytest.raises(ValueError, match="actor/critic/parent"):
        factory.bind(changed)
    changed = copy.deepcopy(policy)
    changed["step_motor_proof"]["decision_start_frame"] = 29
    with pytest.raises(ValueError, match="integrity"):
        factory.bind(changed)
    factory._prototype._policy_hash = "changed"
    with pytest.raises(ValueError, match="prototype"):
        factory.bind(policy)


def test_seed_law_ownership_and_source_drift(learned, monkeypatch):  # noqa: F811
    factory = RecurrentSamplingEpisodeFactory(learned[-1])
    for seed in (True, -1, 2**32, 0.0, "773", None):
        with pytest.raises(ValueError, match="integer"):
            factory.sampling_view(seed=seed)
    first_view, second_view = [factory.sampling_view(seed=seed) for seed in (773, 774)]
    first, second = [factory.bind(factory.preview(view)) for view in (first_view, second_view)]
    assert not np.array_equal(first._noise, second._noise)
    first_view["mean_model"]["parameters"]["head_bias"][0] = 123
    with pytest.raises(ValueError, match="actor/critic/parent"):
        factory.preview(first_view)
    assert factory.sampling_view(seed=773)["mean_model"]["parameters"]["head_bias"][0] != 123
    expected = factory._mean_hash
    factory._mean_hash = "sha256:" + "f" * 64
    with pytest.raises(ValueError, match="prototype"):
        factory.contract()
    factory._mean_hash = expected
    path = next(iter(factory._preview._pins))
    monkeypatch.setitem(factory._preview._pins, path, "sha256:" + "f" * 64)
    with pytest.raises(ValueError, match="dependency"):
        factory.bind(factory.preview(second_view))


@pytest.mark.parametrize("mutation", ["hidden", "progress", "draw", "weights"])
def test_cached_numeric_prototype_drift_refused(learned, mutation):  # noqa: F811
    factory = RecurrentSamplingEpisodeFactory(learned[-1])
    policy = factory.preview(factory.sampling_view(seed=773))
    prototype = factory._prototype
    if mutation == "hidden":
        prototype._recurrent._state[0] = 0.1
    elif mutation == "progress":
        prototype._active_frame = 30
    elif mutation == "draw":
        prototype._last_draw[0] = 0.1
    else:
        # Replacing a read-only parameter array must not bypass integrity.
        layer = prototype._behavior._layers[0]
        layer[0].flags.writeable = True
        layer[0].flat[0] += 0.1
    for action in (
        lambda: factory.bind(policy),
        factory.contract,
        lambda: factory.preview(policy["step_motor_proof"]["model"]),
        lambda: factory.sampling_view(seed=774),
    ):
        with pytest.raises(ValueError, match="prototype"):
            action()
