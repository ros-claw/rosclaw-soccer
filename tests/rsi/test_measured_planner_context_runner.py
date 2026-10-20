"""Admission/proxy fixtures, not physical or real-ONNX qualification."""

import sys

import numpy as np
import pytest

from scripts import rsi_probe_measured_planner_context as audit


def arguments(tmp_path):
    result = ["runner"]
    for name in ("summary", "world", "model-root", "core-source", "output"):
        result += ["--" + name, str(tmp_path / name)]
    result += ["--expected-head", "expected", "--expected-summary-hash", "sha256:" + "1" * 64]
    for i in range(4):
        result += ["--producer", str(tmp_path / f"p{i}")]
    return result


@pytest.mark.parametrize("kind", ["source", "output", "missing_cluster"])
def test_incomplete_admission_before_physics(tmp_path, monkeypatch, kind):
    args = arguments(tmp_path)
    if kind == "missing_cluster":
        args = args[:-2]
    if kind == "output":
        (tmp_path / "output").mkdir()
    monkeypatch.setattr(sys, "argv", args)
    monkeypatch.setattr(
        audit.subprocess,
        "check_output",
        lambda *a, **k: "wrong\n" if kind == "source" else "expected\n",
    )
    with pytest.raises(ValueError, match="sealed source"):
        audit.main()


def test_summary_commitment_required_before_project_imports(tmp_path, monkeypatch):
    (tmp_path / "summary").mkdir()
    (tmp_path / "summary" / "complete.json").write_text("{}")
    monkeypatch.setattr(sys, "argv", arguments(tmp_path))
    monkeypatch.setattr(audit.subprocess, "check_output", lambda *a, **k: "expected\n")
    with pytest.raises(ValueError, match="outcome identity differs"):
        audit.main()
    assert not (tmp_path / "output").exists()


def test_recording_proxy_does_not_replace_shared_methods_and_owns_copies():
    class FixtureSession:
        def run(self, names, feed):
            return [feed["context"], np.array([4], dtype=np.int64)]

    session = FixtureSession()
    original_method = session.run
    proxy = audit.RecordingPlanner(session)
    context = np.ones((1, 4, 36), dtype=np.float32)
    returned = proxy.run(None, {"context": context})
    assert session.run == original_method
    assert returned[0] is context
    assert not np.shares_memory(proxy.feeds[0]["context"], context)
    assert not np.shares_memory(proxy.outputs[0][0], context)
    context[0, 0, 0] = 2
    assert proxy.feeds[0]["context"][0, 0, 0] == 1
    assert proxy.outputs[0][0][0, 0, 0] == 1


def test_proxy_retains_feed_on_model_error():
    class FixtureSession:
        def run(self, names, feed):
            raise ValueError("fixture model error")

    proxy = audit.RecordingPlanner(FixtureSession())
    with pytest.raises(ValueError, match="fixture model error"):
        proxy.run(None, {"context": np.zeros((1, 4, 36), dtype=np.float32)})
    assert len(proxy.feeds) == 1
    assert proxy.outputs == []
