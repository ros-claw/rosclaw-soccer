import json
import sys
from pathlib import Path

import pytest

from scripts import rsi_read_mesh_contact_native_pilot as audit
from scripts import rsi_summarize_contact_exit_mpc_pilot as summary
from tests.rsi.test_release_four_full_terminal_readback import fixture


def setup_summary(tmp_path, monkeypatch, *, directional=False, surface=False):
    audited = fixture(
        tmp_path,
        monkeypatch,
        torque_mpc=not (directional or surface),
        directional=directional,
        surface=surface,
    )
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


def test_directional_summary_preserves_failure_and_does_not_certify_all_body_safety(
    tmp_path, monkeypatch
):
    _, output = setup_summary(tmp_path, monkeypatch, directional=True)
    summary.main()
    result = json.loads(output.read_text())
    assert result["schema"] == "soccer.rsi.directional_compliance_measured_summary.v1"
    assert result["decision"] == "REJECT_LOCAL_TEACHER"
    assert result["rows"][2]["evaluated"]["outcome"]["safe"] is False
    assert result["additional_all_robot_contact_safety_review_required"] is True
    assert result["all_robot_body_contact_safety_confirmed"] is False
    assert result["teacher_qualified"] is False
    assert result["new_optimizer_updates"] == 0


def test_surface_summary_keeps_six_cases_and_original_none_retention(tmp_path, monkeypatch):
    _, output = setup_summary(tmp_path, monkeypatch, surface=True)
    summary.main()
    result = json.loads(output.read_text())
    assert result["schema"] == "soccer.rsi.mesh_surface_measured_summary.v1"
    assert len(result["rows"]) == 6
    assert result["paired_development"]["pairs"] == 6
    assert result["original_none_raw_retention"]["pairs"] == 4
    assert result["source_body_integrations"] == 42000
    assert result["teacher_qualified"] is False
    assert result["all_robot_body_contact_safety_confirmed"] is False
