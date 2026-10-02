"""CLI fixture: full reconstruction output, no optimizer called."""

import json

import numpy as np

from scripts import rsi_fit_parallel_memory_motor as workflow


def test_audit_only_preserves_all_frames_without_running_legacy_optimizer(monkeypatch, tmp_path):
    from rosclaw_soccer.rsi import smooth_memory_learning, smooth_memory_motor
    from scripts import rsi_fit_smooth_memory_motor

    model = tmp_path / "model.json"
    model.write_text(json.dumps({"model_hash": "fixture-model"}))
    summary = dict(
        commitment={"samples_per_course": 4}, rows=[{"index": 0}], report_hash="fixture-summary"
    )
    monkeypatch.setattr(
        workflow, "_sealed", lambda p: summary if p.name == "training_summary.json" else {}
    )
    exploration = tmp_path / "exploration"
    exploration.mkdir()
    (exploration / "commitment.json").write_text("{}")
    monkeypatch.setattr(smooth_memory_motor, "validate_model", lambda m: None)
    monkeypatch.setattr(rsi_fit_smooth_memory_motor, "checked_curriculum", lambda *args: [[42, 0]])

    def forbidden(*args, **kwargs):
        raise AssertionError("audit-only must not optimize")

    monkeypatch.setattr(smooth_memory_learning, "fit_update", forbidden)
    arrays = {
        "observation": np.zeros((1080, 134)),
        "phase_index": np.zeros(1080, dtype=np.int64),
        "latent_action": np.zeros((1080, 12)),
        "old_log_probability": np.zeros(1080),
        "terminal_return": np.zeros(1080),
        "std_raw": np.full(1080, 0.1),
        "trajectory_index": np.repeat(np.arange(4), 270),
    }
    monkeypatch.setattr(
        workflow, "ordered_audits", lambda *args: iter([([{"group": i} for i in range(4)], arrays)])
    )
    output = tmp_path / "audit"
    monkeypatch.setattr(
        "sys.argv",
        [
            "fixture",
            "--behavior-kind",
            "smooth-memory",
            "--audit-only",
            "--exploration-root",
            str(exploration),
            "--parent-bank-root",
            str(tmp_path),
            "--model",
            str(model),
            "--output-root",
            str(output),
        ],
    )
    workflow.main()
    manifest = json.loads((output / "rollout_manifest.json").read_text())
    assert manifest["physical_rollout_count"] == 4
    assert manifest["frame_sample_count"] == 1080
    assert not (output / "model.json").exists()
    with np.load(output / "rollouts.npz", allow_pickle=False) as data:
        for key, value in arrays.items():
            np.testing.assert_array_equal(data[key], value)
