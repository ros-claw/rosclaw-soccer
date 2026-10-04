"""Synthetic neural MC integration; no claim of physical improvement."""

import copy

import numpy as np
import pytest
from rosclaw.growth.context_prediction_mlp import fit_context_predictor

from rosclaw_soccer.rsi.proposal_memory_learning import fit_update
from rosclaw_soccer.rsi.proposal_memory_motor import (
    CompiledProposalMemoryMotor,
    initial_model,
    make_preview,
    validate_model,
)
from rosclaw_soccer.sim.contracts import hash_json
from tests.rsi.test_current_memory_motor import current  # noqa: F401
from tests.rsi.test_kernel_guarded_step_motor import candidate  # noqa: F401
from tests.rsi.test_phase_balanced_proposal_motor import imbalanced_complete_batch
from tests.rsi.test_smooth_memory_motor import smooth_parent  # noqa: F401
from tests.rsi.test_step_motor_network import model  # noqa: F401


def test_neural_mc_receipt_is_complete_and_cannot_grant_motion(current, smooth_parent):  # noqa: F811
    initial = initial_model(
        current[0],
        maximum_mean_kl=0.05,
        loss_weighting_profile="equal-contact-phase-mass",
        critic_profile="whole-context-neural",
    )
    data = imbalanced_complete_batch(smooth_parent)
    decoder = CompiledProposalMemoryMotor(make_preview(initial))
    phi = np.stack([decoder.features(x) for x in data["observation"]])
    x = np.column_stack((phi, data["phase_index"]))
    y = data["terminal_return"][:, None]
    groups = data["trajectory_index"]
    fits = [
        fit_context_predictor(
            x, y, groups, held_out_contexts=(i,), epochs=20, batch_size=128, seed=470 + i
        )
        for i in range(4)
    ]
    learned = fit_update(
        initial,
        data,
        batch_hash="sha256:" + "b" * 64,
        trajectory_context_ids=np.arange(4),
        context_evidence_hash="sha256:" + "c" * 64,
        neural_critic_fit_results=fits,
    )
    validate_model(learned)
    receipt = learned["learning_receipt"]
    assert receipt["critic_kind"] == "WHOLE_CONTEXT_NEURAL_CROSSFIT_MC_NOT_TD_OR_GAE"
    assert receipt["critic_readout_role"] == "LINEAR_DIAGNOSTIC_ONLY_NEURAL_USED_FOR_ADVANTAGES"
    assert len(receipt["critic_model_hashes"]) == 4
    assert receipt["overlapping_context_count"] == 0
    assert receipt["runtime_execution_authorized"] is False
    assert learned["initial_actor"] == initial["initial_actor"]
    assert learned["residual_layers"] != initial["residual_layers"]
    for fault in (
        "critic_hash",
        "authority",
        "partial",
        "leak",
        "source",
        "readout_role",
        "no_update",
    ):
        bad = copy.deepcopy(learned)
        if fault == "critic_hash":
            bad["learning_receipt"]["critic_model_hashes"][0] = "sha256:" + "0" * 64
        elif fault == "authority":
            bad["neural_critic_fit_results"][0]["hardware_authorized"] = True
        elif fault == "partial":
            bad["neural_critic_fit_results"].pop()
        elif fault == "leak":
            bad["neural_critic_fit_results"][0]["held_out_contexts"] = [False]
        elif fault == "source":
            bad["core_prediction_source_hash"] = "sha256:" + "0" * 64
        elif fault == "no_update":
            bad["neural_critic_fit_results"][0]["optimizer_updates"] = 0
        else:
            bad["learning_receipt"]["critic_readout_role"] = "USED_FOR_ACTUAL_VALUE"
        bad.pop("model_hash")
        bad["model_hash"] = hash_json(bad)
        with pytest.raises(ValueError):
            validate_model(bad)
    old = initial_model(current[0], maximum_mean_kl=0.05, critic_profile="whole-context")
    with pytest.raises(ValueError, match="silently alter"):
        fit_update(
            old,
            data,
            batch_hash="sha256:" + "b" * 64,
            trajectory_context_ids=np.arange(4),
            context_evidence_hash="sha256:" + "c" * 64,
            neural_critic_fit_results=fits,
        )
