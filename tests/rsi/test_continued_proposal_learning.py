"""Synthetic sequential updates, explicitly not new physical rollouts."""

import copy

import numpy as np
import pytest

from rosclaw_soccer.rsi.proposal_decoder_selection import select_proposal_decoder
from rosclaw_soccer.rsi.proposal_memory_learning import fit_update
from rosclaw_soccer.rsi.proposal_memory_motor import initial_model, make_preview, validate_model
from rosclaw_soccer.sim.contracts import hash_json
from tests.rsi.test_current_memory_motor import current  # noqa: F401
from tests.rsi.test_kernel_guarded_step_motor import candidate  # noqa: F401
from tests.rsi.test_phase_balanced_proposal_motor import imbalanced_complete_batch
from tests.rsi.test_smooth_memory_motor import smooth_parent  # noqa: F401
from tests.rsi.test_step_motor_network import model  # noqa: F401


def resample_numeric_policy(actor, data):
    """Independent AR(1) simulation of latent actions, not robot physics."""
    decoder = select_proposal_decoder(make_preview(actor), implementation="bounded_snapshot")
    rng = np.random.default_rng(513)
    actions, densities = [], []
    previous_noise = np.zeros(12)
    for i, (x, phase) in enumerate(zip(data["observation"], data["phase_index"], strict=True)):
        reset = i % 270 == 0
        mean = decoder.raw_mean(x, int(phase))
        conditional = mean if reset else mean + 0.9 * previous_noise
        scale = 0.1 if reset else 0.1 * np.sqrt(1 - 0.9**2)
        innovation = scale * rng.normal(size=12)
        action = conditional + innovation
        densities.append(
            float(
                np.sum(-0.5 * (innovation / scale) ** 2 - np.log(scale) - 0.5 * np.log(2 * np.pi))
            )
        )
        actions.append(action)
        previous_noise = action - mean
    return dict(data, latent_action=np.stack(actions), old_log_probability=np.asarray(densities))


def test_two_updates_require_current_behavior_and_preserve_frozen_identity(
    current,  # noqa: F811
    smooth_parent,  # noqa: F811
):
    initial = initial_model(current[0], maximum_mean_kl=0.05)
    old_data = imbalanced_complete_batch(smooth_parent)
    first = fit_update(
        initial,
        old_data,
        batch_hash="sha256:" + "b" * 64,
        numeric_implementation="bounded_snapshot",
    )
    with pytest.raises(ValueError, match="actual immediate behavior"):
        fit_update(first, old_data, batch_hash="sha256:" + "c" * 64)
    with pytest.raises(ValueError, match="new independently bound batch"):
        fit_update(
            first,
            old_data,
            batch_hash="sha256:" + "b" * 64,
            behavior_model_hash=first["model_hash"],
        )
    with pytest.raises(ValueError, match="actual immediate behavior"):
        fit_update(
            first,
            old_data,
            batch_hash="sha256:" + "c" * 64,
            behavior_model_hash=first["learning_receipt"]["behavior_model_hash"],
        )
    with pytest.raises(ValueError, match="actual conditional behavior likelihood"):
        fit_update(
            first,
            old_data,
            batch_hash="sha256:" + "c" * 64,
            behavior_model_hash=first["model_hash"],
            numeric_implementation="bounded_snapshot",
        )
    second = fit_update(
        first,
        resample_numeric_policy(first, old_data),
        batch_hash="sha256:" + "d" * 64,
        behavior_model_hash=first["model_hash"],
        numeric_implementation="bounded_snapshot",
    )
    validate_model(second)
    assert second["generation"] == 2
    assert second["previous_model"] == first
    assert second["initial_actor"] == initial["initial_actor"]
    assert second["residual_layers"] != first["residual_layers"]
    receipt = second["learning_receipt"]
    assert receipt["behavior_model_hash"] == first["model_hash"]
    assert receipt["learner_parent_hash"] == first["model_hash"]
    assert receipt["physical_batch_verified"] is False
    assert receipt["exact_mean_conditional_kl"] <= 0.05
    assert receipt["exact_mean_marginal_kl"] <= 0.05
    for fault in ("missing", "skip", "objective", "behavior", "authority"):
        bad = copy.deepcopy(second)
        if fault == "missing":
            bad.pop("previous_model")
        elif fault == "skip":
            bad["previous_model"] = initial
        elif fault == "objective":
            bad["maximum_mean_kl"] = 0.1
        elif fault == "behavior":
            bad["learning_receipt"]["behavior_model_hash"] = initial["model_hash"]
        else:
            bad["previous_model"]["hardware_authorized"] = True
        bad.pop("model_hash")
        bad["model_hash"] = hash_json(bad)
        with pytest.raises(ValueError):
            validate_model(bad)
