"""Synthetic loss/receipt contracts; no claim of improved physical kicking."""

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


def test_first_contact_lead_receipt_has_no_future_actor_inputs(current, smooth_parent):  # noqa: F811
    initial = initial_model(
        current[0], maximum_mean_kl=0.05, loss_weighting_profile="equal-first-contact-lead-mass"
    )
    learned = fit_update(
        initial,
        imbalanced_complete_batch(smooth_parent),
        batch_hash="sha256:" + "a" * 64,
        measured_event_frames=np.full(4, 59, dtype=np.int64),
        event_evidence_hash="sha256:" + "b" * 64,
    )
    validate_model(learned)
    receipt = learned["learning_receipt"]
    assert receipt["event_partition_frame_counts"] == [84, 36, 80, 880, 0]
    assert receipt["sample_weighting"]["maximum"] == 7.5
    assert receipt["future_event_is_actor_observation"] is False
    assert receipt["physical_batch_verified"] is False
    for key, value in (
        ("future_event_is_actor_observation", True),
        ("credit_lead_frames", 9),
        ("event_evidence_hash", "not-bound"),
    ):
        forged = copy.deepcopy(learned)
        forged["learning_receipt"][key] = value
        forged.pop("model_hash")
        forged["model_hash"] = hash_json(forged)
        with pytest.raises(ValueError, match="event-credit provenance"):
            validate_model(forged)


def test_no_event_failures_are_kept_with_positive_full_frame_weights(current, smooth_parent):  # noqa: F811
    initial = initial_model(
        current[0], maximum_mean_kl=0.05, loss_weighting_profile="equal-first-contact-lead-mass"
    )
    learned = fit_update(
        initial,
        imbalanced_complete_batch(smooth_parent),
        batch_hash="sha256:" + "a" * 64,
        measured_event_frames=np.full(4, -1, dtype=np.int64),
        event_evidence_hash="sha256:" + "b" * 64,
    )
    assert learned["learning_receipt"]["event_partition_frame_counts"] == [0, 0, 0, 0, 1080]
    weights = learned["learning_receipt"]["sample_weighting"]
    assert weights["minimum"] == weights["maximum"] == 1.0
    assert weights["all_numeric_rows_retained"] is True
    validate_model(learned)


@pytest.mark.parametrize(
    "events,evidence",
    [
        (None, None),
        ([59] * 3, "sha256:" + "b" * 64),
        ([59.0] * 4, "sha256:" + "b" * 64),
        ([True] * 4, "sha256:" + "b" * 64),
        ([59] * 4, "not-bound"),
    ],
)
def test_incomplete_or_unbound_event_labels_rejected(
    current,  # noqa: F811
    smooth_parent,  # noqa: F811
    monkeypatch,
    events,
    evidence,  # noqa: F811
):  # noqa: F811
    initial = initial_model(
        current[0], maximum_mean_kl=0.05, loss_weighting_profile="equal-first-contact-lead-mass"
    )
    data = imbalanced_complete_batch(smooth_parent)
    monkeypatch.setattr(
        "rosclaw_soccer.rsi.proposal_memory_learning.CompiledProposalMemoryMotor",
        lambda *_: pytest.fail("must reject before decoder allocation"),
    )
    with pytest.raises(ValueError, match="offline event labels"):
        fit_update(
            initial,
            data,
            batch_hash="sha256:" + "a" * 64,
            measured_event_frames=events,
            event_evidence_hash=evidence,
        )


def test_event_labels_cannot_silently_change_uniform_objective(current, smooth_parent):  # noqa: F811
    initial = initial_model(current[0], maximum_mean_kl=0.05)
    with pytest.raises(ValueError, match="silently alter"):
        fit_update(
            initial,
            imbalanced_complete_batch(smooth_parent),
            batch_hash="sha256:" + "a" * 64,
            measured_event_frames=[59] * 4,
            event_evidence_hash="sha256:" + "b" * 64,
        )
