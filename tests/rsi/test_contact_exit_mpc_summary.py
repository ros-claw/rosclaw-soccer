import json
import sys
from pathlib import Path

import pytest

from scripts import rsi_read_mesh_contact_native_pilot as audit
from scripts import rsi_summarize_contact_exit_mpc_pilot as summary
from tests.rsi.test_release_four_full_terminal_readback import fixture


def setup_summary(tmp_path, monkeypatch):
    audited = fixture(tmp_path, monkeypatch, torque_mpc=True)
    audit.main()
    output = audited.parent / "summary.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "summary",
            "--config",
            str(audited.parent / "config.json"),
            "--audit",
            str(audited),
            "--reader-source",
            str(Path(audit.__file__)),
            "--output",
            str(output),
        ],
    )
    return audited, output


def test_complete_summary_is_not_learning_or_teacher_qualification(tmp_path, monkeypatch):
    _, output = setup_summary(tmp_path, monkeypatch)
    summary.main()
    result = json.loads(output.read_text())
    assert len(result["rows"]) == 4
    assert result["decision"] == "REJECT_LOCAL_TEACHER"
    assert result["new_body_integrations"] == 0
    assert result["new_optimizer_updates"] == 0
    assert result["teacher_qualified"] is False
    assert result["successor_ready"] is None
    assert result["rows"][2]["evaluated"]["outcome"]["safe"] is False


def test_summary_rejects_changed_actual_trace_even_with_existing_terminal_audit(
    tmp_path, monkeypatch
):
    audited, output = setup_summary(tmp_path, monkeypatch)
    complete = json.loads((audited.parent / "complete.json").read_text())
    Path(complete["results"][0]["witnesses"]["trace"]["path"]).write_bytes(b"changed after audit")
    with pytest.raises(ValueError, match="trace changed"):
        summary.main()
    assert not output.exists()
