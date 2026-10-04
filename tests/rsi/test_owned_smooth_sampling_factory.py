import copy

import numpy as np
import pytest

from rosclaw_soccer.rsi.owned_smooth_sampling_factory import OwnedSmoothSamplingFactory
from rosclaw_soccer.rsi.smooth_decoder_selection import (
    select_smooth_decoder,
    validate_sampling_compilation_contract,
)
from rosclaw_soccer.rsi.smooth_memory_motor import (
    CompiledSmoothMemoryMotor,
    make_preview,
    make_sampling_view,
)
from tests.rsi.test_kernel_guarded_step_motor import candidate  # noqa: F401
from tests.rsi.test_smooth_memory_motor import smooth_parent  # noqa: F401
from tests.rsi.test_step_motor_execution import body
from tests.rsi.test_step_motor_network import model  # noqa: F401


def test_all_frames_actions_density_and_episode_state_match_original(smooth_parent):  # noqa: F811
    factory = OwnedSmoothSamplingFactory(smooth_parent)
    view = make_sampling_view(smooth_parent, seed=466, std=0.1)
    policy = factory.preview(view)
    assert policy == make_preview(view)
    decoder, contract = select_smooth_decoder(policy, sampling_factory=factory)
    validate_sampling_compilation_contract(contract, policy)
    assert contract["full_original_preview_validation"] is False
    assert contract["original_mean_semantics_validated_once"] is True
    assert contract["complete_original_preview_integrity_checked_each_bind"] is True
    original = CompiledSmoothMemoryMotor(policy)
    other = factory.bind(policy)
    observations = {k: np.concatenate([v] * 8) for k, v in body().items()}
    previous, forces = np.zeros(12), np.zeros(6)
    for frame in range(300):
        if frame == 70:
            forces[0] = 2
        boundary = dict(
            frame=frame,
            nominal_target=np.zeros(29),
            baseline=np.zeros(12),
            limits=np.tile([-1.0, 1.0], (12, 1)),
            previous=previous,
            previous_contact_forces=forces,
        )
        actual = decoder.delta_at_frame(policy, observations, **boundary)
        expected = original.delta_at_frame(policy, observations, **boundary)
        np.testing.assert_array_equal(actual, expected)
        previous = actual
        if frame >= 30:
            x = np.full(134, frame / 1000)
            a, lp = decoder.latent_sample(x, frame, frame % 3)
            b, lq = original.latent_sample(x, frame, frame % 3)
            np.testing.assert_array_equal(a, b)
            assert lp == lq
    assert other._memory.last_frame == -1
    assert other._parent._memory is not decoder._parent._memory
    assert other._noise is not decoder._noise
    assert not decoder._noise.flags.writeable


@pytest.mark.parametrize("key", ["policy_hash", "hardware_authorized", "execution_profile"])
def test_preview_integrity_is_byte_strict(smooth_parent, key):  # noqa: F811
    factory = OwnedSmoothSamplingFactory(smooth_parent)
    policy = factory.preview(make_sampling_view(smooth_parent, seed=467, std=0.1))
    bad = copy.deepcopy(policy)
    bad[key] = 0 if key == "hardware_authorized" else "wrong"
    with pytest.raises(ValueError, match="integrity"):
        factory.bind(bad)
