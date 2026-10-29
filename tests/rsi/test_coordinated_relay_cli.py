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


def test_child_return_role_rejects_missing_qualified_continuation(tmp_path):
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
            "--receiver-child-return-role-pair",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "child return-role pair requires isolated precontact continuation" in result.stderr
    assert not (tmp_path / "output").exists()


def test_own_sampling_rejects_missing_identity_protocol(tmp_path):
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
            "--return-player-own-sampling",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "own sampling requires the adaptive identity pair" in result.stderr
    assert not (tmp_path / "output").exists()


def test_sampling_seed_without_own_sampling_fails_closed(tmp_path):
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
            "--return-player-sampling-seed",
            "94137",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "explicit sampling seed requires own sampling" in result.stderr
    assert not (tmp_path / "output").exists()


def test_prepared_pair_requires_child_study(tmp_path):
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
            "--return-player-prepared-pair",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "prepared pair requires isolated child return-role study" in result.stderr
    assert not (tmp_path / "output").exists()


def test_return_candidate_rejects_incomplete_lineage(tmp_path):
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
            "--return-player-candidate",
            str(tmp_path / "candidate.pt"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert (
        "return candidate requires isolated prepared adaptive pair and exact evidence"
        in result.stderr
    )
    assert not (tmp_path / "output").exists()


def test_velocity_match_prepared_requires_child_role(tmp_path):
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
            "--receiver-velocity-match-prepared-pair",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "velocity match requires an isolated child return-role pair" in result.stderr
    assert not (tmp_path / "output").exists()


def test_child_sampling_requires_velocity_match_protocol(tmp_path):
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
            "--receiver-child-velocity-sampling",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "child velocity sampling requires prepared velocity-match pair" in result.stderr
    assert not (tmp_path / "output").exists()
