import copy

import numpy as np
import pytest

from rosclaw_soccer.rsi.compiled_step_inference import CompiledStepMotor
from rosclaw_soccer.rsi.memory_guarded_phase_transfer import (
    CompiledMemoryPhaseMotor,
    make_model,
    make_preview,
    validate_model,
)
from rosclaw_soccer.rsi.step_motor_execution import make_preview as warm_preview
from rosclaw_soccer.sim.contracts import hash_json
from tests.rsi.test_kernel_guarded_step_motor import candidate  # noqa: F401
from tests.rsi.test_protected_phase_step_network import protected  # noqa: F401
from tests.rsi.test_step_motor_execution import body
from tests.rsi.test_step_motor_network import model  # noqa: F401


@pytest.fixture(scope="module")
def transfer(candidate, protected):  # noqa: F811
    # Synthetic sealed numeric fixture, NOT physical learning evidence.
    phase = copy.deepcopy(protected[0])
    phase["generation"] = 1
    phase["learning_receipt"] = dict(
        algorithm="PROTECTED_PHASE_PPO_CLIP_MC_TERMINAL",
        physical_batch_hash="sha256:" + "d" * 64,
        frozen_base_encoder=True,
        actor_head_only=True,
        distributional_retention_guaranteed=False,
        promotion_authorized=False,
        hardware_authorized=False,
        exact_mean_latent_kl=0.0,
    )
    phase["model_hash"] = hash_json({k: v for k, v in phase.items() if k != "model_hash"})
    return make_model(phase, candidate[0])


def test_transfer_preserves_lineage_without_claiming_new_optimizer_work(transfer):
    validate_model(transfer)
    assert transfer["new_optimizer_steps"] == 0
    assert transfer["phase_model"]["generation"] == 1
    assert transfer["promotion_authorized"] is False
    assert transfer["hardware_authorized"] is False


def test_zero_inherited_fixture_reproduces_warm_actual_decoder(transfer):
    policy = make_preview(transfer)
    warm = warm_preview(transfer["phase_model"]["base_model"])
    decoder = CompiledMemoryPhaseMotor(policy)
    reference = CompiledStepMotor.from_legacy_preview(warm)
    prior = np.zeros(12)
    observed = body()
    for frame in range(33):
        kwargs = dict(
            frame=frame,
            nominal_target=np.zeros(29),
            baseline=np.zeros(12),
            limits=np.tile([-1.0, 1.0], (12, 1)),
            previous=prior,
            previous_contact_forces=np.zeros(6),
        )
        delta = decoder.delta_at_frame(policy, observed, **kwargs)
        assert np.array_equal(delta, reference.delta_at_frame(warm, observed, **kwargs))
        prior = delta


def test_resealed_transfer_cannot_change_authority_or_invent_optimizer_work(transfer):
    for field, value in (("hardware_authorized", True), ("new_optimizer_steps", 1)):
        changed = copy.deepcopy(transfer)
        changed[field] = value
        changed["model_hash"] = hash_json({k: v for k, v in changed.items() if k != "model_hash"})
        with pytest.raises(ValueError):
            validate_model(changed)
