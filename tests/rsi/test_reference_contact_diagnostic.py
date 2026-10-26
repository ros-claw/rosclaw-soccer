import numpy as np
import pytest
from test_candidate_body23_comparison import _models

from rosclaw_soccer.rsi.reference_contact_diagnostic import (
    contact_label_motion_diagnostic,
    provisional_prior_support,
    reference_ankle_positions,
)
from rosclaw_soccer.rsi.shared_target_forecast import compiled_model_hash


def test_reports_fast_claimed_support_without_relabeling():
    pos = np.zeros((5, 2, 3), dtype=np.float64)
    pos[:, 0, 0] = np.arange(5) * 0.1
    labels = np.ones((5, 2), dtype=bool)
    before = labels.copy()
    result = contact_label_motion_diagnostic(pos, labels, fps=10.0)
    assert result["claimed_contact_fast_fraction"] == pytest.approx(0.5)
    assert result["reference_ankle_center_speed_is_not_contact_point_slip"] is True
    assert result["labels_automatically_changed"] is False
    np.testing.assert_array_equal(labels, before)


def test_rejects_nonfinite_or_unlabeled_trajectories():
    pos = np.zeros((5, 2, 3), dtype=np.float64)
    with pytest.raises(ValueError):
        contact_label_motion_diagnostic(pos, np.zeros((5, 2), dtype=bool), fps=30.0)
    pos[2, 0, 0] = np.nan
    with pytest.raises(ValueError):
        contact_label_motion_diagnostic(pos, np.ones((5, 2), dtype=bool), fps=30.0)


def test_ankles_are_computed_from_scratch_named_body_kinematics():
    model, _ = _models()
    fields = {
        "dof": np.zeros((3, 23), np.float64),
        "root_rot": np.array([[0, 0, 0, 1]] * 3, np.float64),
        "root_trans_offset": np.array([[0, 0, 1], [0.1, 0, 1], [0.2, 0, 1]], np.float64),
    }
    ankles = reference_ankle_positions(
        model, fields, expected_model_hash=compiled_model_hash(model)
    )
    assert ankles.shape == (3, 2, 3)
    np.testing.assert_allclose(ankles[:, 0, 0], [0, 0.1, 0.2], atol=1e-12)
    with pytest.raises(ValueError):
        reference_ankle_positions(model, fields, expected_model_hash="sha256:" + "0" * 64)


def test_provisional_support_uses_only_prior_labels_and_velocity():
    positions = np.zeros((5, 2, 3), np.float64)
    positions[2, 1, 0] = 0.1
    labels = np.array([[True, False], [True, False], [True, True], [False, False], [False, False]])
    result = provisional_prior_support(positions, labels, frame=2, fps=30.0)
    np.testing.assert_array_equal(result["provisional_mask"], [True, False])
    assert result["source_labels_changed"] is False
    assert result["actual_contact_proven"] is False
    assert result["future_frames_used"] is False
    positions[3:] = np.nan
    again = provisional_prior_support(positions, labels, frame=2, fps=30.0)
    np.testing.assert_array_equal(again["provisional_mask"], result["provisional_mask"])


def test_provisional_support_abstains_if_no_stable_foot():
    pos = np.zeros((3, 2, 3), np.float64)
    labels = np.array([[True, False], [False, True], [True, True]])
    result = provisional_prior_support(pos, labels, frame=2, fps=30.0)
    assert result["abstained"] is True
    assert not result["provisional_mask"].any()
