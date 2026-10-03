"""Historical review-path contracts; fixtures are not physical evidence."""

import numpy as np
import pytest

from rosclaw_soccer.sim.contracts import hash_bytes
from scripts import rsi_fit_protected_phase_step_motor as module


def test_explicit_replayed_review_location_keeps_every_frame_and_same_hash(tmp_path, monkeypatch):
    folder = tmp_path / "old-case"
    folder.mkdir()
    trace = folder / "physical_trace.npz"
    np.savez_compressed(
        trace,
        force_n=np.zeros((300, 1, 6)),
        pre_motor_joint_target_rad=np.zeros((300, 1, 29)),
        motor_delta_rad=np.zeros((300, 1, 29)),
    )
    replayed = tmp_path / "new-replayed-review.json"
    raw = dict(report_hash="report", physical_trace_hash=hash_bytes(trace.read_bytes()))
    review = dict(
        report_hash="review",
        reviewed_report_hash="report",
        actual_mujoco_dynamics_replayed=True,
        actual_pd_torque_reconstructed=True,
        neural_target_reconstructed=True,
    )
    seen = []

    def sealed(path):
        seen.append(path)
        return raw if path == folder / "report.json" else review

    monkeypatch.setattr(module, "_sealed", sealed)
    frames = []

    def features(_body, *, frame, **_kwargs):
        frames.append(frame)
        return np.full(134, frame, dtype=float)

    monkeypatch.setattr(module, "features_at_frame", features)
    monkeypatch.setattr(module, "phase_sequence", lambda _forces: np.arange(300) % 3)
    x, phase, reread = module.cpu_features(
        folder, dict(report_hash="report", review_hash="review"), review_path=replayed
    )
    assert seen == [folder / "report.json", replayed]
    assert frames == list(range(30, 300))
    assert x.shape == (270, 134)
    assert np.array_equal(phase, np.arange(300)[30:] % 3)
    assert reread == raw


@pytest.mark.parametrize("explicit", [False, True])
def test_wrong_review_identity_rejected_before_trace_open(tmp_path, monkeypatch, explicit):
    raw = dict(report_hash="report")
    review = dict(report_hash="wrong-review")
    seen = []

    def sealed(path):
        seen.append(path)
        return raw if path == tmp_path / "report.json" else review

    monkeypatch.setattr(module, "_sealed", sealed)
    selected = tmp_path / "explicit.json" if explicit else None
    with pytest.raises(ValueError, match="replay artifact changed"):
        module.cpu_features(
            tmp_path, dict(report_hash="report", review_hash="expected"), review_path=selected
        )
    assert seen[-1] == (selected if explicit else tmp_path / "review.json")
