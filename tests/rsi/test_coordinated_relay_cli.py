"""The paired coordinated protocol must not silently broaden old studies."""

import subprocess
import sys


def test_coordinated_protocol_rejects_incomplete_teacher_configuration(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.rsi_collect_neural_relay_chain",
            "--frontier-root",
            str(tmp_path / "frontier"),
            "--frontier-head",
            "unused",
            "--frontier-hash",
            "unused",
            "--expected-head",
            "unused",
            "--output",
            str(tmp_path / "output"),
            "--new-context-block",
            "0",
            "--receiver-coordinated-intercept",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "requires the isolated early-contact foot teacher pair" in result.stderr
    assert not (tmp_path / "output").exists()
