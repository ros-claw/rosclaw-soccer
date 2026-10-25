import runpy
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "script",
    [
        "rsi_run_native_command_family.py",
        "rsi_train_native_role_command.py",
        "rsi_query_native_role_teacher.py",
        "rsi_initialize_native_role_residual.py",
        "rsi_collect_native_role_residual.py",
        "rsi_fit_native_role_residual_td3.py",
        "rsi_run_native_residual_learning.py",
        "rsi_review_native_role_learning.py",
        "rsi_probe_native_team_intent.py",
        "rsi_run_native_contact_authority.py",
        "rsi_probe_experimental_velocity_envelope.py",
        "rsi_run_accelerated_approach_contact.py",
    ],
)
def test_command_family_bootstrap_does_not_train_or_allocate_physics(script, tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[2]
    monkeypatch.chdir(tmp_path)
    namespace = runpy.run_path(str(root / "scripts" / script), run_name="__mp_main__")
    assert callable(namespace["main"])
    assert not list(tmp_path.iterdir())


def test_learned_physics_cannot_run_without_exact_training_and_clearance_commitment():
    command = [sys.executable, "-m", "scripts.rsi_probe_native_team_intent"]
    for key in (
        "parent",
        "own",
        "core-source-root",
        "output",
        "parent-hash",
        "own-hash",
        "core-expected-head",
        "expected-head",
    ):
        command += ["--" + key, "unused"]
    result = subprocess.run(
        command + ["--command-training", "unused"], capture_output=True, text=True
    )
    assert result.returncode == 2
    assert "learned command contrast requires exact training" in result.stderr


def test_single_player_scope_requires_explicit_residual_before_allocation():
    command = [sys.executable, "-m", "scripts.rsi_probe_native_team_intent"]
    for key in (
        "parent",
        "own",
        "core-source-root",
        "output",
        "parent-hash",
        "own-hash",
        "core-expected-head",
        "expected-head",
    ):
        command += ["--" + key, "unused"]
    result = subprocess.run(
        command + ["--residual-agent", "red_finisher"], capture_output=True, text=True
    )
    assert result.returncode == 2
    assert "single-player residual scope requires an explicit residual head" in result.stderr


def test_experimental_approach_does_not_accept_a_bare_flag_or_unbound_physics():
    command = [sys.executable, "-m", "scripts.rsi_probe_native_team_intent"]
    for key in (
        "parent",
        "own",
        "core-source-root",
        "output",
        "parent-hash",
        "own-hash",
        "core-expected-head",
        "expected-head",
    ):
        command += ["--" + key, "unused"]
    result = subprocess.run(
        command + ["--experimental-approach-config", "unused"], capture_output=True, text=True
    )
    assert result.returncode == 2
    assert "experimental approach requires exact config" in result.stderr
