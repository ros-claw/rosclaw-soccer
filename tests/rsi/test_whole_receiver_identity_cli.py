import os
import subprocess
import sys


def test_collector_and_auditor_are_module_launchable_without_pythonpath_dot():
    env = dict(os.environ, PYTHONPATH="src")
    for module in (
        "scripts.rsi_collect_neural_relay_chain",
        "scripts.rsi_audit_neural_relay_chain",
    ):
        result = subprocess.run(
            [sys.executable, "-m", module, "--help"],
            capture_output=True,
            text=True,
            env=env,
            check=False,
        )
        assert result.returncode == 0, result.stderr


def test_center_curriculum_rejects_missing_identity_before_allocation(tmp_path):
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
            "--receiver-center-curriculum",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "center curriculum requires the receiver identity pair" in result.stderr
    assert not (tmp_path / "out").exists()


def test_precontact_fullbody_rejects_other_interventions_before_allocation(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.rsi_collect_neural_relay_chain",
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
            "--precontact-fullbody-pair",
            "--receiver-whole-identity",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "precontact full-body pair requires one isolated" in result.stderr
    assert not (tmp_path / "out").exists()


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


def test_center_exploration_without_identity_proof_fails_before_allocation(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.rsi_collect_neural_relay_chain",
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
            "--receiver-center-curriculum",
            "--receiver-exploration",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "exact identity review" in result.stderr
    assert not (tmp_path / "out").exists()
