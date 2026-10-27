import subprocess
import sys


def test_identity_rejects_mixed_intervention_before_allocation(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "scripts/rsi_collect_neural_relay_chain.py",
            "--frontier-root",
            str(tmp_path / "frontier"),
            "--frontier-head",
            "head",
            "--frontier-hash",
            "hash",
            "--expected-head",
            "head",
            "--output",
            str(tmp_path / "out"),
            "--new-context-block",
            "0",
            "--receiver-whole-identity",
            "--receiver-follow-study",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "requires one isolated development block" in result.stderr
    assert not (tmp_path / "out").exists()


def test_exploration_without_identity_proof_fails_before_allocation(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "scripts/rsi_collect_neural_relay_chain.py",
            "--frontier-root",
            str(tmp_path / "frontier"),
            "--frontier-head",
            "head",
            "--frontier-hash",
            "hash",
            "--expected-head",
            "head",
            "--output",
            str(tmp_path / "out"),
            "--new-context-block",
            "0",
            "--receiver-exploration",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "exact identity review" in result.stderr
    assert not (tmp_path / "out").exists()
