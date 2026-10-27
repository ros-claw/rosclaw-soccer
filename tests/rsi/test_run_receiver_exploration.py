import json
from types import SimpleNamespace

import pytest

from scripts import rsi_run_receiver_exploration as run


def test_rejected_identity_never_allocates_physics(monkeypatch, tmp_path):
    monkeypatch.setattr(run, "clean", lambda _: None)
    review = tmp_path / "review.json"
    review.write_text(json.dumps({"status": "RELAY_WHOLE_RECEIVER_ZERO_IDENTITY_REJECTED"}))
    monkeypatch.setattr(
        run.sys,
        "argv",
        [
            "run",
            "--frontier-root",
            str(tmp_path / "frontier"),
            "--frontier-head",
            "h",
            "--frontier-hash",
            "f",
            "--identity-review",
            str(review),
            "--identity-unit",
            "rosclaw-team-identity.service",
            "--identity-review-head",
            "h",
            "--new-context-block",
            "0",
            "--output",
            str(tmp_path / "output"),
            "--expected-head",
            "h",
        ],
    )
    calls = []

    def fake(cmd, **_kwargs):
        calls.append(cmd)
        return SimpleNamespace(
            returncode=0, stdout="ActiveState=inactive\nResult=success\nExecMainStatus=0\n"
        )

    monkeypatch.setattr(run.subprocess, "run", fake)
    with pytest.raises(ValueError, match="strict full zero identity"):
        run.main()
    assert len(calls) == 1
    assert not (tmp_path / "output").exists()
