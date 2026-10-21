import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts import rsi_continue_paid_contact as pipeline
from scripts.rsi_continue_paid_contact import checked_receipt


def test_closed_receipt_commits_bytes(tmp_path):
    (tmp_path / "complete.json").write_text(json.dumps({"status": "PASS"}))
    report, commitment = checked_receipt(tmp_path, "PASS")
    assert report == {"status": "PASS"}
    assert commitment.startswith("sha256:") and len(commitment) == 71


def test_failure_overrides_completion(tmp_path):
    (tmp_path / "complete.json").write_text('{"status":"PASS"}')
    (tmp_path / "failed.json").write_text("{}")
    with pytest.raises(ValueError, match="without a failure"):
        checked_receipt(tmp_path, "PASS")


def test_wrong_status_never_advances(tmp_path):
    (tmp_path / "complete.json").write_text('{"status":"PENDING"}')
    with pytest.raises(ValueError, match="status"):
        checked_receipt(tmp_path, "PASS")


def test_symlink_receipt_rejected(tmp_path):
    (tmp_path / "other.json").write_text('{"status":"PASS"}')
    (tmp_path / "complete.json").symlink_to(tmp_path / "other.json")
    with pytest.raises(ValueError, match="ordinary"):
        checked_receipt(tmp_path, "PASS")


@pytest.mark.parametrize("failure", [None, "child_exit", "wrong_receipt"])
def test_pipeline_advances_only_on_actual_checked_stage(tmp_path, monkeypatch, failure):
    exam, primitive, neural = [tmp_path / n for n in ("exam", "primitive", "neural")]
    for root in (exam, primitive, neural):
        root.mkdir()
    (exam / "complete.json").write_text(
        json.dumps(
            dict(
                status="PAID_LIVE_DISTILLED_CPU_FULL_WORLD_EXAM_CLOSED",
                actual_neural_rows=42,
                candidates=[dict(full_continuous_skill_gate=True)],
            )
        )
    )
    _, eh = checked_receipt(exam, "PAID_LIVE_DISTILLED_CPU_FULL_WORLD_EXAM_CLOSED")
    (primitive / "complete.json").write_text(
        json.dumps(
            dict(
                status="PAID_LIVE_DISTILLED_ALL_PRIMITIVE_AUDIT_PASS",
                input_file_hashes={str(exam / "complete.json"): eh},
            )
        )
    )
    _, ph = checked_receipt(primitive, "PAID_LIVE_DISTILLED_ALL_PRIMITIVE_AUDIT_PASS")
    (neural / "complete.json").write_text(
        json.dumps(
            dict(
                status="PAID_GATED_CONTACT_ALL_NEURAL_PROPOSALS_EXACT",
                own_neural_rows=42,
                input_file_hashes={
                    str(exam / "complete.json"): eh,
                    str(primitive / "complete.json"): ph,
                },
            )
        )
    )
    _, nh = checked_receipt(neural, "PAID_GATED_CONTACT_ALL_NEURAL_PROPOSALS_EXACT")
    output = tmp_path / "pipeline"
    args = ["pipeline", "--expected-head", "frozen", "--output", str(output)]
    for name, root, h in (("exam", exam, eh), ("primitive", primitive, ph), ("neural", neural, nh)):
        args += ["--" + name + "-root", str(root), "--" + name + "-hash", h]
    monkeypatch.setattr(sys, "argv", args)
    monkeypatch.setattr(subprocess, "check_output", lambda args, **kw: "frozen\n" if kw else b"")
    calls = []

    def run(args, *, check):
        assert check
        calls.append(args)
        if failure == "child_exit":
            raise subprocess.CalledProcessError(1, args)
        root = Path(args[args.index("--output") + 1])
        root.mkdir()
        status = (
            "PAID_CONTINUING_CHAIN_CPU_EXAM_CLOSED_PENDING_INDEPENDENT_AUDIT"
            if len(calls) == 1
            else "PAID_CONTINUING_CHAIN_NATIVE_AND_NEURAL_AUDIT_PASS"
        )
        (root / "complete.json").write_text(
            json.dumps(
                dict(
                    status="WRONG" if failure == "wrong_receipt" else status,
                )
            )
        )

    monkeypatch.setattr(subprocess, "run", run)
    if failure:
        with pytest.raises((ValueError, subprocess.CalledProcessError)):
            pipeline.main()
        assert len(calls) == 1  # Never retry, nor run audit after unverified physics.
        assert (output / "failed.json").is_file()
        assert not (output / "complete.json").exists()
    else:
        pipeline.main()
        assert len(calls) == 2
        assert (output / "physics_closed.json").is_file()
        report = json.loads((output / "complete.json").read_text())
        assert report["goal_complete"] is False
        assert report["hardware_authorized"] is False
