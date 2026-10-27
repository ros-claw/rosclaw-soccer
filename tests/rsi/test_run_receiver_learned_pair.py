import sys

from scripts import rsi_run_receiver_learned_pair as runner


def test_runner_invokes_package_modules_for_child_imports(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(runner, "clean", lambda _: None)
    monkeypatch.setattr(runner, "digest", lambda _: "sha256:test")
    monkeypatch.setattr(runner.subprocess, "run", lambda command, check: calls.append(command))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run",
            "--frontier-root",
            "frontier",
            "--frontier-head",
            "f",
            "--frontier-hash",
            "h",
            "--receiver-checkpoint",
            "checkpoint",
            "--receiver-checkpoint-hash",
            "h",
            "--receiver-candidate-audit",
            "audit",
            "--receiver-candidate-audit-hash",
            "h",
            "--receiver-dataset",
            "dataset",
            "--receiver-dataset-hash",
            "h",
            "--new-context-block",
            "0",
            "--expected-head",
            "h",
            "--output",
            str(tmp_path / "output"),
        ],
    )
    runner.main()
    assert calls[0][1:3] == ["-m", "scripts.rsi_collect_neural_relay_chain"]
    assert calls[1][1:3] == ["-m", "scripts.rsi_audit_neural_relay_chain"]
