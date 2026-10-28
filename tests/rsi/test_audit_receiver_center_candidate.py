import sys

import pytest

from scripts import rsi_audit_receiver_candidate as audit


def test_center_candidate_requires_independent_data_audit_before_model_load(monkeypatch, tmp_path):
    monkeypatch.setattr(audit, "clean", lambda _: None)
    monkeypatch.setattr(audit, "digest", lambda _: "pinned")
    for name in ("dataset", "admission", "training"):
        folder = tmp_path / name
        folder.mkdir()
        (folder / "complete.json").write_text("{}")
    output = tmp_path / "out"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "audit",
            "--dataset",
            str(tmp_path / "dataset"),
            "--admission",
            str(tmp_path / "admission"),
            "--training",
            str(tmp_path / "training"),
            "--output",
            str(output),
            "--dataset-hash",
            "pinned",
            "--admission-hash",
            "pinned",
            "--training-hash",
            "pinned",
            "--producer-head",
            "head",
            "--expected-head",
            "head",
            "--receiver-center-strict",
        ],
    )
    with pytest.raises(ValueError, match="exact audited data and seed"):
        audit.main()
    assert not output.exists()
