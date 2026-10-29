import numpy as np
import pytest

from scripts import rsi_review_receiver_transition_pair as review


def test_target_jump_measures_actual_executed_targets(monkeypatch, tmp_path):
    target = np.zeros((14, 29))
    target[3:, 0] = 0.1
    target[4:, 0] = 0.14
    p = tmp_path / "physics.npz"
    np.savez(p, target_b=target)
    monkeypatch.setattr(review, "bind", lambda *args: p)
    result = review.transition_target_metrics(tmp_path, {"handoff": {"control": 3}})
    assert result["first_target_jump_rad"] == pytest.approx(0.1)
    assert result["transition_max_target_jump_rad"] == pytest.approx(0.1)
    assert result["observed_transition_controls"] == 10


def test_no_handoff_cannot_hide_a_different_physical_history():
    base = {
        "handoff": None,
        "artifacts": {
            name: {"hash": "same"}
            for name in ("initial", "physics", "receiver_velocity", "receiver_shadow")
        },
    }
    candidate = {
        "handoff": None,
        "artifacts": {name: dict(item) for name, item in base["artifacts"].items()},
    }
    assert review.verify_before_handoff(None, base, candidate) is None
    candidate["artifacts"]["physics"]["hash"] = "different"
    with pytest.raises(ValueError):
        review.verify_before_handoff(None, base, candidate)
