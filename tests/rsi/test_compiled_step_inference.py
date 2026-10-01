import copy

import numpy as np
import pytest

from rosclaw_soccer.rsi.compiled_step_inference import CompiledStepMotor, make_model, make_preview
from rosclaw_soccer.rsi.step_motor_execution import delta_at_frame as original_delta
from rosclaw_soccer.rsi.step_motor_execution import make_preview as original_preview
from rosclaw_soccer.rsi.stochastic_step_execution import delta_at_frame as sampled_delta
from rosclaw_soccer.rsi.stochastic_step_execution import latent_sample, make_sampling_view
from rosclaw_soccer.rsi.stochastic_step_execution import make_preview as sampled_preview
from rosclaw_soccer.sim.contracts import hash_json
from tests.rsi.test_step_motor_execution import body
from tests.rsi.test_step_motor_network import model  # noqa: F401


@pytest.mark.parametrize("sampling", [False, True])
def test_compilation_preserves_measured_actions_without_rehashing_every_frame(model, sampling):  # noqa: F811
    base = make_sampling_view(model, seed=319, std=0.1) if sampling else model
    original = sampled_preview(base) if sampling else original_preview(base)
    reference = sampled_delta if sampling else original_delta
    wrapped = make_model(base)
    policy = make_preview(wrapped)
    compiled = CompiledStepMotor(policy)
    legacy_compiled = CompiledStepMotor.from_legacy_preview(original)
    observation = body()
    previous = np.zeros(12)
    for frame in (20, 30, 31, 32, 33):
        boundary = dict(
            frame=frame,
            nominal_target=np.zeros(29),
            baseline=np.zeros(12),
            limits=np.tile([-1.0, 1.0], (12, 1)),
            previous=previous,
            previous_contact_forces=np.zeros(6),
        )
        actual = compiled.delta_at_frame(policy, observation, **boundary)
        expected = reference(original, observation, **boundary)
        assert np.array_equal(actual, expected)
        assert np.array_equal(
            actual, legacy_compiled.delta_at_frame(original, observation, **boundary)
        )
        previous = actual
    # The running decoder cannot observe mutation of the input model after sealing.
    wrapped["base_model"]["actor"]["layers"][0]["bias"][0] = 10000
    assert np.array_equal(actual, compiled.delta_at_frame(policy, observation, **boundary))
    with pytest.raises(ValueError):
        CompiledStepMotor(policy)
    if sampling:
        value, logp = legacy_compiled.latent_sample(np.zeros(134), 30)
        reference_value, reference_logp = latent_sample(base, np.zeros(134), 30)
        assert np.array_equal(value, reference_value) and logp == reference_logp


def test_compiled_policy_cannot_reseal_hardware_authority(model):  # noqa: F811
    wrapped = copy.deepcopy(make_model(model))
    wrapped["hardware_authorized"] = True
    wrapped["model_hash"] = hash_json({k: v for k, v in wrapped.items() if k != "model_hash"})
    with pytest.raises(ValueError):
        make_preview(wrapped)
