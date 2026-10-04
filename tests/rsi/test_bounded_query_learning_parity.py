"""Actual optimizer parity on synthetic data, not physical skill evidence."""

import copy

import numpy as np
import pytest

from rosclaw_soccer.rsi.proposal_memory_learning import fit_update
from rosclaw_soccer.rsi.proposal_memory_motor import initial_model, validate_model
from rosclaw_soccer.sim.contracts import hash_json
from tests.rsi.test_correlated_gradient_numerical_parity import conditional_fixture_batch
from tests.rsi.test_current_memory_motor import current  # noqa: F401
from tests.rsi.test_kernel_guarded_step_motor import candidate  # noqa: F401
from tests.rsi.test_smooth_memory_motor import smooth_parent  # noqa: F401
from tests.rsi.test_step_motor_network import model  # noqa: F401


def test_acceleration_keeps_actual_optimizer_weights_losses_and_kl(current, smooth_parent):  # noqa: F811
    initial = initial_model(
        current[0], maximum_mean_kl=0.05, loss_weighting_profile="equal-contact-phase-mass"
    )
    batch = conditional_fixture_batch(smooth_parent)
    before = copy.deepcopy(initial)
    reference = fit_update(initial, batch, batch_hash="sha256:" + "a" * 64)
    fast = fit_update(
        initial, batch, batch_hash="sha256:" + "a" * 64, numeric_implementation="bounded_snapshot"
    )
    assert initial == before
    assert reference["residual_layers"] == fast["residual_layers"]
    np.testing.assert_array_equal(reference["critic_readout"], fast["critic_readout"])
    a, b = copy.deepcopy(reference["learning_receipt"]), copy.deepcopy(fast["learning_receipt"])
    contract = b.pop("numeric_preparation")
    assert contract["implementation"] == "bounded_snapshot"
    assert contract["hardware_authorized"] is False
    assert a == b
    validate_model(fast)
    bad = copy.deepcopy(fast)
    bad["learning_receipt"]["numeric_preparation"]["core_query_source_hash"] = "sha256:" + "f" * 64
    bad.pop("model_hash")
    bad["model_hash"] = hash_json(bad)
    with pytest.raises(ValueError, match="compilation contract"):
        validate_model(bad)


@pytest.mark.parametrize("implementation", [None, True, "fast", "hardware"])
def test_invalid_learning_selection_fails_before_any_actor_or_optimizer(implementation):
    with pytest.raises(ValueError, match="known proposal"):
        fit_update({}, {}, batch_hash="sha256:" + "a" * 64, numeric_implementation=implementation)
