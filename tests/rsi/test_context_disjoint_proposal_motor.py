"""Declared offline critic contracts, not physical performance evidence."""

import copy

import numpy as np
import pytest

from rosclaw_soccer.rsi.proposal_memory_learning import fit_update
from rosclaw_soccer.rsi.proposal_memory_motor import initial_model, validate_model
from rosclaw_soccer.sim.contracts import hash_json
from tests.rsi.test_current_memory_motor import current  # noqa: F401
from tests.rsi.test_kernel_guarded_step_motor import candidate  # noqa: F401
from tests.rsi.test_phase_balanced_proposal_motor import imbalanced_complete_batch
from tests.rsi.test_smooth_memory_motor import smooth_parent  # noqa: F401
from tests.rsi.test_step_motor_network import model  # noqa: F401


def test_context_critic_receipt_is_bound_and_cannot_become_old_objective(
    current,  # noqa: F811
    smooth_parent,  # noqa: F811
):
    initial = initial_model(
        current[0],
        maximum_mean_kl=0.05,
        loss_weighting_profile="equal-contact-phase-mass",
        critic_profile="whole-context",
    )
    learned = fit_update(
        initial,
        imbalanced_complete_batch(smooth_parent),
        batch_hash="sha256:" + "b" * 64,
        trajectory_context_ids=np.arange(4),
        context_evidence_hash="sha256:" + "c" * 64,
    )
    validate_model(learned)
    receipt = learned["learning_receipt"]
    assert receipt["independent_critic_contexts"] == 4
    assert receipt["overlapping_context_count"] == 0
    assert receipt["context_is_actor_observation"] is False
    assert receipt["critic_readout_unit"] == "raw_terminal_return"
    assert receipt["physical_batch_verified"] is False
    for field, value in (
        ("context_is_actor_observation", True),
        ("overlapping_context_count", False),
        ("independent_critic_contexts", True),
        ("context_labels_hash", "unbound"),
        ("critic_readout_unit", "standardized"),
    ):
        forged = copy.deepcopy(learned)
        forged["learning_receipt"][field] = value
        forged.pop("model_hash")
        forged["model_hash"] = hash_json(forged)
        with pytest.raises(ValueError, match="context-disjoint critic provenance"):
            validate_model(forged)
    forged = copy.deepcopy(learned)
    forged.pop("critic_profile")
    forged.pop("model_hash")
    forged["model_hash"] = hash_json(forged)
    with pytest.raises(ValueError, match="regression receipt"):
        validate_model(forged)


@pytest.mark.parametrize("profile", [None, True, "choose-best-split"])
def test_unknown_critic_profile_rejected(current, profile):  # noqa: F811
    with pytest.raises(ValueError):
        initial_model(current[0], maximum_mean_kl=0.05, critic_profile=profile)


def test_old_profile_cannot_silently_take_context_labels(current, smooth_parent):  # noqa: F811
    initial = initial_model(current[0], maximum_mean_kl=0.05)
    with pytest.raises(ValueError, match="silently alter"):
        fit_update(
            initial,
            imbalanced_complete_batch(smooth_parent),
            batch_hash="sha256:" + "b" * 64,
            trajectory_context_ids=np.arange(4),
            context_evidence_hash="sha256:" + "c" * 64,
        )


def test_new_profile_rejects_unbound_labels(current, smooth_parent):  # noqa: F811
    initial = initial_model(current[0], maximum_mean_kl=0.05, critic_profile="whole-context")
    with pytest.raises(ValueError, match="bound complete context"):
        fit_update(
            initial,
            imbalanced_complete_batch(smooth_parent),
            batch_hash="sha256:" + "b" * 64,
            trajectory_context_ids=np.arange(4),
        )
