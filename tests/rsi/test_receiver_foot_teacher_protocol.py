"""The foot teacher cannot silently compose with other research interventions."""

import subprocess
import sys

import pytest


@pytest.mark.parametrize(
    "conflict",
    [
        ["--receiver-lateral-authority"],
        ["--receiver-follow-study"],
        ["--receiver-whole-identity"],
        ["--passer-residual-pair"],
        ["--precontact-fullbody-pair"],
        ["--reference-bridge-controls", "5"],
    ],
)
def test_foot_teacher_pair_rejects_mixed_experimental_protocol(conflict):
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.rsi_collect_neural_relay_chain",
            "--frontier-root",
            "/not-read",
            "--frontier-head",
            "unused",
            "--frontier-hash",
            "unused",
            "--expected-head",
            "unused",
            "--output",
            "/not-written",
            "--new-context-block",
            "0",
            "--receiver-foot-teacher-pair",
            *conflict,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "foot teacher pair requires one isolated development block" in result.stderr
