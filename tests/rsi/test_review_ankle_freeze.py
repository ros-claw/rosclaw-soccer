import numpy as np
import pytest

from scripts import rsi_review_ankle_freeze as review


def test_teacher_review_checks_actual_frozen_ankles(monkeypatch, tmp_path):
    physics = tmp_path / "physics.npz"
    trace = tmp_path / "trace.npz"
    target = np.zeros((2, 29))
    delta = np.zeros((2, 29))
    delta[1, 0] = 0.02
    np.savez(physics, target_b=target)
    np.savez(
        trace,
        control=np.arange(2),
        FOOT_applied_delta=delta,
        FOOT_target=target,
        FOOT_delta_jump_rad=np.array([0, 0.02]),
        FOOT_active=np.ones(2, bool),
    )
    monkeypatch.setattr(
        review, "bind", lambda root, row, key: physics if key == "physics" else trace
    )
    result = review.teacher_metrics(tmp_path, {}, frozen=True)
    assert result["actual_nonzero_controls"] == 1
    assert result["slew_exceeding_controls"] == 0
    delta[1, 4] = 0.01
    np.savez(
        trace,
        control=np.arange(2),
        FOOT_applied_delta=delta,
        FOOT_target=target,
        FOOT_delta_jump_rad=np.array([0, 0.02]),
        FOOT_active=np.ones(2, bool),
    )
    with pytest.raises(ValueError, match="ankle"):
        review.teacher_metrics(tmp_path, {}, frozen=True)


def test_review_rejects_target_that_was_not_executed(monkeypatch, tmp_path):
    physics = tmp_path / "physics.npz"
    trace = tmp_path / "trace.npz"
    np.savez(physics, target_b=np.ones((1, 29)))
    np.savez(
        trace,
        control=np.arange(1),
        FOOT_applied_delta=np.zeros((1, 29)),
        FOOT_target=np.zeros((1, 29)),
        FOOT_delta_jump_rad=np.zeros(1),
        FOOT_active=np.zeros(1, bool),
    )
    monkeypatch.setattr(
        review, "bind", lambda root, row, key: physics if key == "physics" else trace
    )
    with pytest.raises(ValueError, match="executed teacher targets"):
        review.teacher_metrics(tmp_path, {}, frozen=True)
