"""Synthetic fixtures, not physical actor growth evidence."""

import copy

import numpy as np
import pytest

from rosclaw_soccer.rsi.current_memory_motor import CompiledCurrentMemoryMotor
from rosclaw_soccer.rsi.current_memory_motor import make_preview as current_preview
from rosclaw_soccer.rsi.proposal_memory_learning import fit_update
from rosclaw_soccer.rsi.proposal_memory_motor import (
    CompiledProposalMemoryMotor,
    initial_model,
    make_preview,
    validate_model,
)
from rosclaw_soccer.sim.contracts import hash_json
from tests.rsi.test_correlated_gradient_numerical_parity import conditional_fixture_batch
from tests.rsi.test_current_memory_motor import current  # noqa: F401
from tests.rsi.test_kernel_guarded_step_motor import candidate  # noqa: F401
from tests.rsi.test_smooth_memory_motor import smooth_parent  # noqa: F401
from tests.rsi.test_step_motor_network import model  # noqa: F401


def test_zero_regression_is_globally_identical_to_current_nn(current):  # noqa: F811
    initial, extra = current
    proposal = initial_model(initial, maximum_mean_kl=0.05)
    new = CompiledProposalMemoryMotor(make_preview(proposal))
    old = CompiledCurrentMemoryMotor(current_preview(initial))
    for phase in range(3):
        for x in (extra, np.zeros(134), np.ones(134), np.full(134, -0.19)):
            assert np.array_equal(new.raw_mean(x, phase), old.raw_mean(x, phase))
    assert proposal["maximum_mean_kl"] == 0.05
    assert proposal["runtime_execution_authorized"] is False
    assert new._parent._output_memory.to_dict() == old._parent._output_memory.to_dict()


def test_actual_numeric_regression_changes_layers_preserves_memory_and_guard(
    current,  # noqa: F811
    smooth_parent,  # noqa: F811
):  # noqa: F811
    initial, extra = current
    proposal = initial_model(initial, maximum_mean_kl=0.05)
    learned = fit_update(
        proposal, conditional_fixture_batch(smooth_parent), batch_hash="sha256:" + "b" * 64
    )
    new, old = (CompiledProposalMemoryMotor(make_preview(v)) for v in (learned, proposal))
    assert learned["residual_layers"] != proposal["residual_layers"]
    assert learned["initial_actor"] == proposal["initial_actor"]
    assert np.array_equal(new.raw_mean(extra, 1), old.raw_mean(extra, 1))
    assert learned["learning_receipt"]["physical_batch_verified"] is False
    assert (
        learned["learning_receipt"]["critic_kind"] == "WHOLE_TRAJECTORY_CROSSFIT_MC_NOT_TD_LAMBDA"
    )
    assert learned["learning_receipt"]["maximum_mean_kl"] == 0.05
    assert learned["learning_receipt"]["runtime_execution_authorized"] is False
    with pytest.raises(ValueError, match="zero-addition"):
        fit_update(
            learned, conditional_fixture_batch(smooth_parent), batch_hash="sha256:" + "b" * 64
        )
    for key, value in (
        ("protected_memory_rows", 999),
        ("behavior_model_hash", "sha256:" + "e" * 64),
    ):
        forged = copy.deepcopy(learned)
        forged["learning_receipt"][key] = value
        forged.pop("model_hash")
        forged["model_hash"] = hash_json(forged)
        with pytest.raises(ValueError, match="receipt"):
            validate_model(forged)


@pytest.mark.parametrize(
    "key,value",
    [
        ("raw_residual_cap", 0.3),
        ("learning_rate", 0.01),
        ("generation", True),
        ("hardware_authorized", True),
        ("promotion_authorized", True),
        ("runtime_execution_authorized", True),
        ("proposal_execution_ceiling", "REAL"),
        ("maximum_mean_kl", True),
        ("maximum_mean_kl", 0.501),
        ("memory_query_representation", "DROP_DUPLICATE_EVIDENCE"),
    ],
)
def test_resealed_authority_or_capacity_expansion_rejected(current, key, value):  # noqa: F811
    proposal = initial_model(current[0], maximum_mean_kl=0.05)
    proposal[key] = value
    proposal.pop("model_hash")
    proposal["model_hash"] = hash_json(proposal)
    with pytest.raises(ValueError):
        validate_model(proposal)


def test_new_proposal_is_not_misrepresented_as_a_legacy_approved_model(current):  # noqa: F811
    from rosclaw_soccer.rsi.advantage_memory_motor import validate_model as legacy_validate

    proposal = initial_model(current[0], maximum_mean_kl=0.05)
    with pytest.raises(ValueError, match="sealed bounded"):
        legacy_validate(proposal)
    assert make_preview(proposal)["step_motor_proof"]["qualification"] == (
        "UNQUALIFIED_SIM_PROPOSAL_REGRESSION"
    )


def test_budget_is_explicit_and_committed_in_the_parent_identity(current):  # noqa: F811
    actor = current[0]
    with pytest.raises(TypeError):
        initial_model(actor)
    small = initial_model(actor, maximum_mean_kl=0.005)
    larger = initial_model(actor, maximum_mean_kl=0.05)
    assert small["model_hash"] != larger["model_hash"]
    assert small["residual_layers"] == larger["residual_layers"]
