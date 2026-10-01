import copy

import numpy as np
import pytest

from rosclaw_soccer.rsi.compiled_step_inference import CompiledStepMotor
from rosclaw_soccer.rsi.protected_phase_step_execution import (
    CompiledProtectedPhaseMotor,
    make_preview,
)
from rosclaw_soccer.rsi.protected_phase_step_network import initial_model, latents, validate_model
from rosclaw_soccer.rsi.step_motor_execution import make_preview as warm_preview
from rosclaw_soccer.sim.contracts import hash_json
from tests.rsi.test_step_motor_execution import body
from tests.rsi.test_step_motor_network import model  # noqa: F401


@pytest.fixture(scope="module")
def protected(model):  # noqa: F811
    rng = np.random.default_rng(327)
    anchors = rng.normal(size=(540, 134))
    phases = np.repeat(np.arange(3), 180)
    return (
        initial_model(model, anchors, phases, anchor_bank_hash="sha256:" + "a" * 64),
        anchors,
        phases,
    )


def test_every_anchor_is_protected_even_with_arbitrary_head(protected):
    candidate, anchors, phases = protected
    planes = validate_model(candidate)
    phi = latents(candidate, anchors)
    head = np.random.default_rng(4).normal(size=(3, 12, 512))
    for i, phase in enumerate(phases):
        assert np.array_equal(planes[phase].project(phi[i]), np.zeros(512))
        assert np.array_equal(head[phase] @ planes[phase].project(phi[i]), np.zeros(12))
    assert all(p.plastic_dimensions > 0 for p in planes)


def test_zero_head_is_bit_identical_to_warm_actor(protected):
    candidate, _, _ = protected
    policy = make_preview(candidate)
    warm = warm_preview(candidate["base_model"])
    decoder = CompiledProtectedPhaseMotor(policy)
    reference = CompiledStepMotor.from_legacy_preview(warm)
    previous = np.zeros(12)
    observation = body()
    for frame in range(34):
        kwargs = dict(
            frame=frame,
            nominal_target=np.zeros(29),
            baseline=np.zeros(12),
            limits=np.tile([-1.0, 1.0], (12, 1)),
            previous=previous,
            previous_contact_forces=np.zeros(6),
        )
        actual = decoder.delta_at_frame(policy, observation, **kwargs)
        assert np.array_equal(actual, reference.delta_at_frame(warm, observation, **kwargs))
        previous = actual


@pytest.mark.parametrize(
    "field", ["physics_qualified", "promotion_authorized", "hardware_authorized"]
)
def test_resealed_authority_rejected(protected, field):
    changed = copy.deepcopy(protected[0])
    changed[field] = True
    changed["model_hash"] = hash_json({k: v for k, v in changed.items() if k != "model_hash"})
    with pytest.raises(ValueError):
        validate_model(changed)
