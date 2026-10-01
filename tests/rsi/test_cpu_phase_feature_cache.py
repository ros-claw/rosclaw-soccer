from collections import Counter

import numpy as np

from rosclaw_soccer.sim.contracts import hash_bytes
from scripts import rsi_fit_protected_phase_step_motor as reader


def test_cpu_numeric_members_are_read_once_without_changing_frame_inputs(tmp_path, monkeypatch):
    path = tmp_path / "physical_trace.npz"
    arrays = dict(
        force_n=np.zeros((300, 1, 6)),
        pre_motor_joint_target_rad=np.arange(300 * 29).reshape(300, 1, 29),
        motor_delta_rad=np.arange(300 * 12).reshape(300, 1, 12),
        unused_numeric=np.ones((300, 1)),
    )
    arrays["force_n"][60, 0, 0] = 2
    np.savez_compressed(path, **arrays)
    raw = dict(report_hash="report", physical_trace_hash=hash_bytes(path.read_bytes()))
    reviewed = dict(
        report_hash="review",
        reviewed_report_hash="report",
        actual_mujoco_dynamics_replayed=True,
        actual_pd_torque_reconstructed=True,
        neural_target_reconstructed=True,
    )
    monkeypatch.setattr(reader, "_sealed", lambda p: raw if p.name == "report.json" else reviewed)
    original_load = np.load
    reads = Counter()

    class CountedArchive:
        def __enter__(self):
            self.data = original_load(path, allow_pickle=False)
            self.files = self.data.files
            return self

        def __getitem__(self, key):
            reads[key] += 1
            return self.data[key]

        def __exit__(self, *args):
            self.data.close()

    monkeypatch.setattr(reader.np, "load", lambda *args, **kwargs: CountedArchive())
    frames = []

    def feature(body, *, frame, nominal_target, previous, previous_contact_forces):
        assert isinstance(body, dict)
        assert np.array_equal(nominal_target, arrays["pre_motor_joint_target_rad"][frame, 0])
        assert np.array_equal(previous, arrays["motor_delta_rad"][frame - 1, 0])
        assert np.array_equal(previous_contact_forces, arrays["force_n"][frame - 1, 0])
        frames.append(frame)
        return np.full(134, frame)

    monkeypatch.setattr(reader, "features_at_frame", feature)
    x, phase, result = reader.cpu_features(
        tmp_path, dict(report_hash="report", review_hash="review")
    )
    assert frames == list(range(30, 300))
    assert reads == Counter(dict.fromkeys(arrays, 1))
    assert x.shape == (270, 134)
    assert np.array_equal(phase, reader.phase_sequence(arrays["force_n"][:, 0])[30:])
    assert result is raw
