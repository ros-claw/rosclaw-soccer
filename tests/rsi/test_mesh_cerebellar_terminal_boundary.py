"""Ensure unfinished jobs cannot be mistaken for terminal audit evidence."""

import sys

import pytest

from scripts import rsi_read_mesh_cerebellar_study as reader


@pytest.mark.parametrize(
    "status",
    [
        "MainPID=23\nActiveState=active\nResult=success\n",
        "MainPID=0\nActiveState=failed\nResult=exit-code\n",
        "MainPID=0\nActiveState=inactive\nResult=signal\n",
    ],
)
def test_nonterminal_or_failed_workers_rejected_before_reading_inputs(
    monkeypatch, tmp_path, status
):
    output = tmp_path / "not-created.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "reader",
            "--config",
            str(tmp_path / "absent.json"),
            "--unit",
            "synthetic.service",
            "--output",
            str(output),
        ],
    )
    monkeypatch.setattr(reader.subprocess, "check_output", lambda *args, **kwargs: status)
    with pytest.raises(ValueError, match="terminal"):
        reader.main()
    assert not output.exists()


def test_existing_audit_is_never_overwritten(monkeypatch, tmp_path):
    output = tmp_path / "preserved.json"
    output.write_text("original")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "reader",
            "--config",
            str(tmp_path / "absent.json"),
            "--unit",
            "synthetic.service",
            "--output",
            str(output),
        ],
    )
    monkeypatch.setattr(
        reader.subprocess,
        "check_output",
        lambda *args, **kwargs: "MainPID=0\nActiveState=inactive\nResult=success\n",
    )
    with pytest.raises(ValueError, match="unused"):
        reader.main()
    assert output.read_text() == "original"


def test_incomplete_prior_audit_is_not_implicitly_retried(monkeypatch, tmp_path):
    output = tmp_path / "result.json"
    started = output.with_suffix(".started.json")
    started.write_text("preserved failed/unfinished audit")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "reader",
            "--config",
            str(tmp_path / "absent.json"),
            "--unit",
            "synthetic.service",
            "--output",
            str(output),
        ],
    )
    monkeypatch.setattr(
        reader.subprocess,
        "check_output",
        lambda *args, **kwargs: "MainPID=0\nActiveState=inactive\nResult=success\n",
    )
    with pytest.raises(ValueError, match="unused"):
        reader.main()
    assert not output.exists()
    assert started.read_text() == "preserved failed/unfinished audit"
