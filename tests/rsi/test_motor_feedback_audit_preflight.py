import sys

import pytest

from scripts import rsi_audit_motor_feedback as audit


def test_wrong_audit_source_rejected_before_network_creation(tmp_path, monkeypatch):
    args = ["audit"]
    for key in ("bank", "train-root", "model-root", "output"):
        args += ["--" + key, str(tmp_path / key)]
    for key in ("expected-head", "expected-bank-hash", "expected-training-hash"):
        args += ["--" + key, "declared"]
    monkeypatch.setattr(sys, "argv", args)
    monkeypatch.setattr(audit.subprocess, "check_output", lambda *a, **kw: "wrong-head\n")
    with pytest.raises(ValueError, match="clean frozen"):
        audit.main()
    assert not (tmp_path / "output").exists()
