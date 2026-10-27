import sys
from types import SimpleNamespace

import pytest

from scripts import rsi_watch_receiver_learned_pair as watcher


@pytest.mark.parametrize("result", ["success", "exit-code"])
def test_review_only_after_three_successful_loaded_workers(tmp_path, monkeypatch, result):
    calls = []
    monkeypatch.setattr(watcher, "clean", lambda _: None)
    args = [
        "watch",
        "--source-head",
        "a",
        "--old-head",
        "b",
        "--expected-head",
        "c",
        "--output",
        str(tmp_path / "review"),
    ]
    for i in range(3):
        args.extend(
            [
                "--group",
                f"root{i}",
                f"rosclaw-receiver-learned-pair-recovery-b{i}.service",
                f"old{i}",
                f"old_audit{i}",
            ]
        )
    monkeypatch.setattr(sys, "argv", args)

    def run(command, **kwargs):
        calls.append(command)
        if command[0] == "systemctl":
            code = "0" if result == "success" else "1"
            state = f"LoadState=loaded\nActiveState=inactive\nResult={result}\n"
            return SimpleNamespace(returncode=0, stdout=state + f"ExecMainStatus={code}\n")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(watcher.subprocess, "run", run)
    if result == "success":
        watcher.main()
        assert calls[-1][1:3] == ["-m", "scripts.rsi_review_receiver_learned_pair"]
        assert len(calls) == 4
    else:
        with pytest.raises(ValueError, match="failed"):
            watcher.main()
        assert len(calls) == 1
