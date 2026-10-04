import copy

import numpy as np
import pytest

from rosclaw_soccer.rsi.smooth_decoder_selection import (
    select_smooth_decoder,
    validate_sampling_compilation_contract,
)
from rosclaw_soccer.rsi.smooth_memory_motor import make_preview, make_sampling_view
from rosclaw_soccer.rsi.smooth_sampling_decoder_factory import SmoothSamplingDecoderFactory
from rosclaw_soccer.sim.contracts import hash_json
from tests.rsi.test_kernel_guarded_step_motor import candidate  # noqa: F401
from tests.rsi.test_smooth_memory_motor import smooth_parent  # noqa: F401
from tests.rsi.test_step_motor_network import model  # noqa: F401


def test_default_and_shared_selection_preserve_density_and_own_state(smooth_parent):  # noqa: F811
    policy = make_preview(make_sampling_view(smooth_parent, seed=453, std=0.1))
    factory = SmoothSamplingDecoderFactory(smooth_parent)
    original, absent = select_smooth_decoder(policy)
    fast, contract = select_smooth_decoder(policy, sampling_factory=factory)
    second, _ = select_smooth_decoder(policy, sampling_factory=factory)
    assert absent is None
    validate_sampling_compilation_contract(contract, policy)
    for frame in (30, 70, 299):
        x = np.full(134, frame / 1000)
        a, lp = fast.latent_sample(x, frame, frame % 3)
        b, lq = original.latent_sample(x, frame, frame % 3)
        np.testing.assert_array_equal(a, b)
        assert lp == lq
    assert fast._memory is not second._memory
    assert fast._parent._memory is not second._parent._memory
    assert fast._noise is not second._noise


@pytest.mark.parametrize("factory", [object(), lambda _: None])
def test_unverified_factory_rejected(factory):
    with pytest.raises(ValueError, match="verified"):
        select_smooth_decoder({}, sampling_factory=factory)


@pytest.mark.parametrize("fault", ["mean", "authority", "factory_source_hash", "extra"])
def test_sampling_compilation_contract_is_exact(smooth_parent, fault):  # noqa: F811
    policy = make_preview(make_sampling_view(smooth_parent, seed=454, std=0.1))
    factory = SmoothSamplingDecoderFactory(smooth_parent)
    _, contract = select_smooth_decoder(policy, sampling_factory=factory)
    forged = copy.deepcopy(contract)
    if fault == "mean":
        forged["mean_model_hash"] = "sha256:" + "a" * 64
    elif fault == "authority":
        forged["hardware_authorized"] = True
    elif fault == "extra":
        forged["unknown"] = 0
    else:
        forged[fault] = "sha256:" + "b" * 64
    with pytest.raises(ValueError, match="provenance"):
        validate_sampling_compilation_contract(forged, policy)


def test_factory_cannot_accept_regular_mean_or_resealed_other_mean(smooth_parent):  # noqa: F811
    factory = SmoothSamplingDecoderFactory(smooth_parent)
    with pytest.raises(ValueError):
        select_smooth_decoder(make_preview(smooth_parent), sampling_factory=factory)
    other = copy.deepcopy(smooth_parent)
    other["residual_layers"][0]["bias"][0] += 0.1
    other.pop("model_hash")
    other["model_hash"] = hash_json(other)
    policy = make_preview(make_sampling_view(other, seed=455, std=0.1))
    with pytest.raises(ValueError):
        select_smooth_decoder(policy, sampling_factory=factory)


@pytest.mark.parametrize(
    "policy",
    [
        None,
        [],
        {},
        {"step_motor_proof": None},
        {"step_motor_proof": {"model": {"mean_model": None}}},
    ],
)
def test_malformed_compilation_policy_rejected(policy):
    with pytest.raises(ValueError):
        validate_sampling_compilation_contract({}, policy)
