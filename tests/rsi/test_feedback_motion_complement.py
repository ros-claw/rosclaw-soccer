import numpy as np
import pytest

from rosclaw_soccer.rsi.feedback_motion_complement import fit_feedback_complement


def test_learned_complement_recovers_new_successful_directions_without_overwriting_old():
    rng = np.random.default_rng(9)
    backbone = np.eye(6, 29, dtype=np.float64)
    old = backbone.copy()
    signals = rng.normal(size=(200, 12))
    deltas = np.zeros((200, 29), np.float64)
    deltas[:, :12] = signals
    basis, report = fit_feedback_complement(deltas, backbone)
    np.testing.assert_array_equal(backbone, old)
    assert abs(basis[:, :6]).max() < 1e-10
    assert np.linalg.matrix_rank(basis[:, 6:12]) == 6
    assert report["outside_backbone_energy_fraction"] > 0.3
    assert not report["frozen_backbone_modified"] and not report["physics_qualified"]
    assert report["neural_optimizer_updates"] == 0


def test_no_invented_axes_when_feedback_has_no_new_rank():
    backbone = np.eye(6, 29, dtype=np.float64)
    deltas = np.zeros((200, 29), np.float64)
    deltas[:, :6] = 1
    with pytest.raises(ValueError, match="rank"):
        fit_feedback_complement(deltas, backbone)


def test_nonfinite_feedback_and_degenerate_backbone_rejected():
    backbone = np.eye(6, 29, dtype=np.float64)
    bad = np.zeros((100, 29), np.float64)
    bad[0, 0] = np.nan
    with pytest.raises(ValueError, match="observed"):
        fit_feedback_complement(bad, backbone)
    with pytest.raises(ValueError, match="rank6"):
        fit_feedback_complement(np.ones((100, 29)), np.zeros((6, 29)))
