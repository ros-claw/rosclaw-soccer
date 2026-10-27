import numpy as np
import pytest

from rosclaw_soccer.rsi.receiving_handoff_diagnosis import diagnose_no_handoff


def trajectory():
    q = np.zeros((300, 10, 79), dtype=np.float64)
    v = np.zeros((300, 10, 76), dtype=np.float64)
    q[:, :, 2] = 0.8
    q[:, :, 3] = 1
    q[:, :, 36] = 0.5
    q[:, :, 38] = 0.11
    q[:, :, 39] = 1
    f = np.zeros((300, 10, 2, 3), dtype=np.float64)
    failed = np.zeros((300, 2), dtype=bool)
    return q, v, f, failed


def run(q, v, f, failed, excluded_contact=False):
    return diagnose_no_handoff(
        q,
        v,
        f,
        failed,
        receiver_q=0,
        receiver_v=0,
        ball_q=36,
        ball_v=35,
        excluded_contact=excluded_contact,
    )


def test_contact_threshold_and_original_consecutive_predicate():
    q, v, f, failed = trajectory()
    f[40, 2, 0, 0] = 11
    f[285, 4, 1, 0] = 1.1
    result = run(q, v, f, failed)
    assert result["first_passer_foot_control"] == 40
    assert result["first_receiver_contact_control"] == 285
    assert result["best_consecutive_candidate_controls"] == 15
    assert not result["no_handoff_reconstruction_consistent"]
    f[285, 4, 1, 0] = 1
    assert not run(q, v, f, failed)["actual_receiver_contact_over_1n"]


def test_excluded_contact_does_not_claim_exact_reconstruction():
    q, v, f, failed = trajectory()
    f[1, 0, 1, 2] = 2
    result = run(q, v, f, failed, excluded_contact=True)
    assert result["excluded_contact_ever"]
    assert result["predicate_reconstruction_exact"] is False


def test_reject_nonfinite_or_wrong_shapes():
    q, v, f, failed = trajectory()
    q[0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        run(q, v, f, failed)
