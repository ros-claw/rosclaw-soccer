import sys

import pytest

from scripts.rsi_probe_live_neural_transition import main


@pytest.mark.parametrize(
    "aim,flags",
    [
        ("nan", ["--contact-release", "--single-player-only"]),
        ("inf", ["--contact-release", "--single-player-only"]),
        ("1.21", ["--contact-release", "--single-player-only"]),
        ("0.75", ["--contact-release"]),
        ("0", ["--single-player-only"]),
    ],
)
def test_aim_protocol_rejected_before_source_read_or_physical_allocation(
    tmp_path, monkeypatch, aim, flags
):
    output = tmp_path / "must-not-be-created"
    arguments = [
        "probe",
        "--source",
        str(tmp_path / "missing-source"),
        "--own",
        str(tmp_path / "missing-own"),
        "--output",
        str(output),
        "--source-hash",
        "not-read",
        "--own-hash",
        "not-read",
        "--expected-head",
        "not-read",
        "--aim-y=" + aim,
    ]
    monkeypatch.setattr(sys, "argv", arguments + flags)
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
    assert not output.exists()
