import numpy as np
import pytest

from scripts import rsi_review_relay_balance_counterfactual as review


def test_teacher_prefix_must_equal_complete_actual_parent(monkeypatch, tmp_path):
    left, right = tmp_path / "left.npz", tmp_path / "right.npz"
    np.savez(left, sample=np.arange(5))
    np.savez(right, sample=np.arange(5))
    handoff = dict(
        control=3, boundary=dict(hash="same"), return_goal=[0, 0, 0], kick_origin=[0, 0, 0]
    )
    base = dict(handoff=handoff, artifacts=dict(initial=dict(hash="same")), index=0)
    candidate = dict(handoff=handoff, artifacts=dict(initial=dict(hash="same")), index=1)
    monkeypatch.setattr(
        review, "bind", lambda _root, row, _name: left if row["index"] == 0 else right
    )
    review.verify_same_handoff(tmp_path, base, candidate)
    np.savez(right, sample=np.array([0, 1, 9, 3, 4]))
    with pytest.raises(ValueError, match="prefix differs"):
        review.verify_same_handoff(tmp_path, base, candidate)
