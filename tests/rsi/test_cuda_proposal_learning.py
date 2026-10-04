import copy

import pytest

from rosclaw_soccer.rsi.proposal_memory_learning import fit_update
from rosclaw_soccer.rsi.proposal_memory_motor import initial_model, validate_model
from rosclaw_soccer.sim.contracts import hash_json
from tests.rsi.test_correlated_gradient_numerical_parity import conditional_fixture_batch
from tests.rsi.test_current_memory_motor import current  # noqa: F401
from tests.rsi.test_kernel_guarded_step_motor import candidate  # noqa: F401
from tests.rsi.test_smooth_memory_motor import smooth_parent  # noqa: F401
from tests.rsi.test_step_motor_network import model  # noqa: F401


def test_actual_gpu_update_has_device_bound_receipt_and_rejects_device_migration(
    current,  # noqa: F811
    smooth_parent,  # noqa: F811
    monkeypatch,
):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("actual CUDA unavailable")
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    initial = initial_model(
        current[0],
        maximum_mean_kl=0.05,
        loss_weighting_profile="equal-contact-phase-mass",
        optimizer_compute_device="cuda:0",
    )
    assert initial["optimizer_compute_device"] == "cuda:0"
    learned = fit_update(
        initial,
        conditional_fixture_batch(smooth_parent),
        batch_hash="sha256:" + "a" * 64,
        numeric_implementation="bounded_snapshot",
    )
    validate_model(learned)
    assert learned["learning_receipt"]["compute_device"] == "cuda:0"
    assert learned["learning_receipt"]["cross_device_bit_identity_claimed"] is False
    assert learned["runtime_execution_authorized"] is False
    for fault in ("device", "identity"):
        bad = copy.deepcopy(learned)
        if fault == "device":
            bad["learning_receipt"]["compute_device"] = "cuda:1"
        else:
            bad["learning_receipt"]["cross_device_bit_identity_claimed"] = True
        bad.pop("model_hash")
        bad["model_hash"] = hash_json(bad)
        with pytest.raises(ValueError, match="CUDA optimizer receipt"):
            validate_model(bad)
