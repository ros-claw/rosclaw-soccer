import subprocess
import sys

import pytest


@pytest.mark.parametrize(
    "options",
    [
        ["--torque-exploration-seed=1"],
        ["--torque-exploration-scale=.25"],
        ["--torque-exploration-seed=-1", "--torque-exploration-scale=.25"],
        ["--torque-exploration-seed=4294967296", "--torque-exploration-scale=.25"],
        ["--torque-exploration-seed=1", "--torque-exploration-scale=nan"],
        ["--torque-exploration-seed=1", "--torque-exploration-scale=.26"],
    ],
)
def test_invalid_exploration_rejected_before_core_or_physics(tmp_path, options):
    output = tmp_path / "must-not-exist"
    args = [
        sys.executable,
        "-m",
        "scripts.rsi_probe_live_neural_transition",
        "--source",
        "missing",
        "--own",
        "missing",
        "--output",
        str(output),
        "--source-hash",
        "unused",
        "--own-hash",
        "unused",
        "--expected-head",
        "unused",
        "--torque-candidate-root",
        "missing",
        "--torque-candidate-hash",
        "unused",
        "--core-source-root",
        "missing",
        "--core-expected-head",
        "unused",
        "--capture-torque-proposals",
        *options,
    ]
    result = subprocess.run(args, capture_output=True, text=True, timeout=10)
    assert result.returncode == 2
    assert "bounded seed/scale" in result.stderr
    assert not output.exists()
