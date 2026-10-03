import copy

import numpy as np
import pytest

from rosclaw_soccer.rsi.smooth_memory_motor import (
    CompiledSmoothMemoryMotor,
    make_preview,
    make_sampling_view,
)
from rosclaw_soccer.rsi.smooth_sampling_decoder_factory import SmoothSamplingDecoderFactory
from tests.rsi.test_kernel_guarded_step_motor import candidate  # noqa: F401
from tests.rsi.test_smooth_memory_motor import smooth_parent  # noqa: F401
from tests.rsi.test_step_motor_execution import body
from tests.rsi.test_step_motor_network import model  # noqa: F401


@pytest.mark.parametrize("seed,rho", [(4, 0), (4, 0.9), (42, 0.9), (999, 0.95)])
def test_full_episode_actions_and_conditional_density_exact(smooth_parent, seed, rho):  # noqa: F811
    factory = SmoothSamplingDecoderFactory(smooth_parent)
    view = make_sampling_view(smooth_parent, seed=seed, std=0.1, rho=rho)
    policy = make_preview(view)
    fast, original = factory.bind(policy), CompiledSmoothMemoryMotor(policy)
    observations = {k: np.concatenate([v] * 8) for k, v in body().items()}
    previous = np.zeros(12)
    forces = np.zeros(6)
    for frame in range(300):
        if frame == 70:
            forces[0] = 2.0
        boundary = dict(
            frame=frame,
            nominal_target=np.zeros(29),
            baseline=np.zeros(12),
            limits=np.tile([-1.0, 1.0], (12, 1)),
            previous=previous,
            previous_contact_forces=forces,
        )
        actual = fast.delta_at_frame(policy, observations, **boundary)
        expected = original.delta_at_frame(policy, observations, **boundary)
        np.testing.assert_array_equal(actual, expected)
        previous = actual
        if frame >= 30:
            x = np.full(134, frame / 1000)
            a, lp = fast.latent_sample(x, frame, frame % 3)
            b, lq = original.latent_sample(x, frame, frame % 3)
            np.testing.assert_array_equal(a, b)
            assert lp == lq


def test_episode_state_is_not_shared_and_seed_input_cannot_mutate_bound_decoder(smooth_parent):  # noqa: F811
    factory = SmoothSamplingDecoderFactory(smooth_parent)
    view = make_sampling_view(smooth_parent, seed=4, std=0.1)
    policy = make_preview(view)
    a, b = factory.bind(policy), factory.bind(policy)
    a._memory.advance(0, np.zeros(6))
    assert b._memory.last_frame == -1
    assert factory._prototype._memory.last_frame == -1
    assert a._parent._memory is not b._parent._memory
    assert a._layers is not b._layers
    assert a._parent._layers is not b._parent._layers
    assert a._parent._residual_layers is not b._parent._residual_layers
    assert a._parent._warm is not b._parent._warm
    assert a._parent._warm.layers is not b._parent._warm.layers
    assert a._noise is not b._noise
    assert not a._noise.flags.writeable
    expected = a.latent_sample(np.zeros(134), 30, 0)
    view["seed"] = 123
    view["std_raw"] = 0.15
    observed = a.latent_sample(np.zeros(134), 30, 0)
    np.testing.assert_array_equal(expected[0], observed[0])
    assert expected[1] == observed[1]


def test_replacing_one_decoder_layer_container_cannot_change_the_next_episode(smooth_parent):  # noqa: F811
    factory = SmoothSamplingDecoderFactory(smooth_parent)
    policy = make_preview(make_sampling_view(smooth_parent, seed=4, std=0.1))
    first = factory.bind(policy)
    expected = first.raw_mean(np.zeros(134), 1)
    weight, bias = first._parent._warm.layers[-1]
    first._parent._warm.layers[-1] = (np.zeros_like(weight), np.full_like(bias, 10))
    assert not np.array_equal(first.raw_mean(np.zeros(134), 1), expected)
    next_episode = factory.bind(policy)
    np.testing.assert_array_equal(next_episode.raw_mean(np.zeros(134), 1), expected)


@pytest.mark.parametrize("fault", ["policy-hash", "view-seed", "mean", "authority"])
def test_complete_preview_validation_not_bypassed(smooth_parent, fault):  # noqa: F811
    factory = SmoothSamplingDecoderFactory(smooth_parent)
    policy = make_preview(make_sampling_view(smooth_parent, seed=4, std=0.1))
    bad = copy.deepcopy(policy)
    if fault == "policy-hash":
        bad["policy_hash"] = "sha256:" + "0" * 64
    elif fault == "view-seed":
        bad["step_motor_proof"]["model"]["seed"] = 123
    elif fault == "mean":
        bad["step_motor_proof"]["model"]["mean_model"]["generation"] = 31
    else:
        bad["step_motor_proof"]["model"]["hardware_authorized"] = True
    with pytest.raises(ValueError):
        factory.bind(bad)


def test_factory_only_accepts_non_sampling_mean(smooth_parent):  # noqa: F811
    with pytest.raises(ValueError, match="ordinary"):
        SmoothSamplingDecoderFactory(make_sampling_view(smooth_parent, seed=4, std=0.1))
