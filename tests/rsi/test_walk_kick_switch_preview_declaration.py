"""Reject bad declarations before context loading or physics allocation."""

import hashlib
import json

import pytest

from scripts import rsi_native_context_recovery_common as provenance
from scripts import rsi_native_fresh_common as sources
from scripts import rsi_probe_walk_kick_switch as runner
from scripts import rsi_teacher_physics_context as context


def digest(path):
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize(
    "defect", [None, "pair", "course", "baseline-pass", "teacher-pass", "path"]
)
def test_public_failures_are_pinned_and_validated_before_loading(tmp_path, monkeypatch, defect):
    proof = tmp_path / "proof.json"
    proof.write_text(
        json.dumps(
            dict(status="CURRENT_WALK_TO_KICK_NN_AND_NATIVE_FUTURE_EXACT", input_file_hashes={})
        )
    )
    raw = tmp_path / "B.npz"
    raw.write_bytes(b"not loaded before context admission")
    baseline = dict(
        arm="B",
        entry_control=63,
        course=[2.0, 0.1, -0.5],
        parameters={"reference_frame": None},
        outcome=dict(task_pass=defect == "baseline-pass"),
        experimental_fast_approach=False,
        artifact=dict(path="../B.npz" if defect == "path" else "B.npz", hash=digest(raw)),
    )
    teacher = dict(baseline, arm="C", outcome=dict(task_pass=defect == "teacher-pass"))
    if defect == "course":
        teacher["course"] = [3.0, 0.1, -0.5]
    b, c = tmp_path / "B.json", tmp_path / "C.json"
    b.write_text(json.dumps(baseline))
    c.write_text(json.dumps(teacher))
    args = [
        "probe",
        "--composed-complete",
        str(proof),
        "--composed-hash",
        digest(proof),
        "--reference-record",
        str(b),
        "--reference-hash",
        digest(b),
        "--failed-teacher-record",
        str(c),
        "--expected-head",
        "pinned",
        "--output",
        str(tmp_path / "output"),
    ]
    if defect != "pair":
        args += ["--failed-teacher-hash", digest(c)]
    monkeypatch.setattr("sys.argv", args)
    monkeypatch.setattr(sources, "clean", lambda head: None)
    monkeypatch.setattr(sources, "source_pins", lambda pins: None)
    monkeypatch.setattr(provenance, "check_pins", lambda pins: None)
    loaded = []

    def load(pins):
        loaded.append(dict(pins))
        raise RuntimeError("context admission reached, no allocation")

    monkeypatch.setattr(context, "load_walk_kick_context", load)
    with pytest.raises(RuntimeError if defect is None else ValueError):
        runner.main()
    assert not (tmp_path / "output").exists()
    if defect is None:
        assert loaded[0][str(b)] == digest(b)
        assert loaded[0][str(c)] == digest(c)
        assert loaded[0][str(raw)] == digest(raw)
    else:
        assert not loaded
