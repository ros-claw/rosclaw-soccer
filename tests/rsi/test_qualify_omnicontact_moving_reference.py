import sys

import pytest

from scripts import rsi_qualify_omnicontact_native as runner


def test_moving_initial_ball_requires_derived_receipt(monkeypatch, tmp_path):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "runner",
            "--external-source",
            str(tmp_path),
            "--external-head",
            "head",
            "--body-source",
            str(tmp_path / "body"),
            "--body-source-hash",
            "hash",
            "--expected-head",
            "head",
            "--output",
            str(tmp_path / "output"),
            "--reference-ball-initial-velocity",
        ],
    )
    with pytest.raises(SystemExit):
        runner.main()
    assert not (tmp_path / "output").exists()


def test_phase_change_requires_explicit_derived_reference(monkeypatch, tmp_path):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "runner",
            "--external-source",
            str(tmp_path),
            "--external-head",
            "head",
            "--body-source",
            str(tmp_path / "body"),
            "--body-source-hash",
            "hash",
            "--expected-head",
            "head",
            "--output",
            str(tmp_path / "output"),
            "--reference-phase-start",
            "30",
        ],
    )
    with pytest.raises(SystemExit):
        runner.main()
    assert not (tmp_path / "output").exists()
