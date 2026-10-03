"""Synthetic learning contracts; not football performance evidence."""

import copy

import numpy as np
import pytest

from rosclaw_soccer.rsi.proposal_memory_learning import fit_update
from rosclaw_soccer.rsi.proposal_memory_motor import initial_model, validate_model
from rosclaw_soccer.rsi.smooth_memory_motor import (
    CompiledSmoothMemoryMotor,
    make_preview,
    make_sampling_view,
)
from rosclaw_soccer.sim.contracts import hash_json
from tests.rsi.test_current_memory_motor import current  # noqa: F401
from tests.rsi.test_kernel_guarded_step_motor import candidate  # noqa: F401
from tests.rsi.test_smooth_memory_motor import smooth_parent  # noqa: F401
from tests.rsi.test_step_motor_network import model  # noqa: F401


def imbalanced_complete_batch(parent):
    n = 4 * 270
    x = np.random.default_rng(348).normal(size=(n, 134))
    phase = np.tile(np.repeat(np.arange(3), [30, 20, 220]), 4)
    actions, probabilities = [], []
    for g in range(4):
        sampled = CompiledSmoothMemoryMotor(
            make_preview(make_sampling_view(parent, seed=348 + g, std=0.1, rho=0.9))
        )
        for f in range(30, 300):
            i = g * 270 + f - 30
            action, probability = sampled.latent_sample(x[i], f, int(phase[i]))
            actions.append(action)
            probabilities.append(probability)
    return dict(
        observation=x,
        phase_index=phase,
        latent_action=np.stack(actions),
        old_log_probability=np.asarray(probabilities),
        terminal_return=np.repeat([-2.0, 3.0, -2.0, 3.0], 270),
        std_raw=np.full(n, 0.1),
        trajectory_index=np.repeat(np.arange(4), 270),
    )


def test_balanced_receipt_retains_all_rows_and_cannot_migrate_to_uniform_parent(
    current,  # noqa: F811
    smooth_parent,  # noqa: F811
):
    initial, _ = current
    proposal = initial_model(
        initial, maximum_mean_kl=0.05, loss_weighting_profile="equal-contact-phase-mass"
    )
    data = imbalanced_complete_batch(smooth_parent)
    learned = fit_update(proposal, data, batch_hash="sha256:" + "b" * 64)
    validate_model(learned)
    receipt = learned["learning_receipt"]
    assert receipt["phase_frame_counts"] == [120, 80, 880]
    weights = receipt["sample_weighting"]
    assert weights["row_count"] == 1080
    assert weights["maximum"] == 4.5
    assert weights["all_numeric_rows_retained"] is True
    assert receipt["physical_batch_verified"] is False
    for target, value in (
        ("row_count", 1079),
        ("minimum", 0.0),
        ("hardware_authorized", True),
        ("mean", 1.1),
    ):
        forged = copy.deepcopy(learned)
        forged["learning_receipt"]["sample_weighting"][target] = value
        forged.pop("model_hash")
        forged["model_hash"] = hash_json(forged)
        with pytest.raises(ValueError, match="weighting receipt"):
            validate_model(forged)
    forged = copy.deepcopy(learned)
    forged["loss_weighting_profile"] = "uniform-frame"
    forged["learning_receipt"]["loss_weighting_profile"] = "uniform-frame"
    forged["learning_receipt"].pop("sample_weighting")
    forged.pop("model_hash")
    forged["model_hash"] = hash_json(forged)
    with pytest.raises(ValueError, match="learner parent"):
        validate_model(forged)


@pytest.mark.parametrize("profile", [None, True, "skip-failures"])
def test_unknown_weighting_profile_rejected_before_learning(current, profile):  # noqa: F811
    with pytest.raises(ValueError, match="sealed bounded"):
        initial_model(current[0], maximum_mean_kl=0.05, loss_weighting_profile=profile)
