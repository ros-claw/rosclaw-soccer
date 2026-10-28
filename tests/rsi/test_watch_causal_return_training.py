import pytest

from scripts import rsi_watch_causal_return_training as watch


def configure(monkeypatch, tmp_path):
    banks = [tmp_path / "b1" / "source", tmp_path / "b2" / "source"]
    for bank in banks:
        bank.mkdir(parents=True)
        audit = bank.parent / "audit" / "complete.json"
        audit.parent.mkdir()
        audit.write_text("{}")
    source_hash = "sha256:" + "a" * 64
    argv = [
        "watch",
        "--trial",
        str(tmp_path / "trial"),
        "--trial-hash",
        "trial",
        "--scene",
        str(tmp_path / "scene"),
        "--scene-hash",
        "scene",
        "--baseline-source",
        str(tmp_path / "baseline"),
        "--baseline-audit",
        str(tmp_path / "baseline-audit"),
        "--baseline-audit-hash",
        "audit",
        "--expected-head",
        "head",
        "--output",
        str(tmp_path / "output"),
    ]
    for i, bank in enumerate(banks, 1):
        argv.extend(
            (
                "--bank",
                str(bank),
                f"rosclaw-relay-lateral-replay-b{i}-recovery-20261028.service",
                source_hash,
            )
        )
    monkeypatch.setattr(watch.sys, "argv", argv)
    monkeypatch.setattr(watch, "clean", lambda _: None)
    monkeypatch.setattr(watch, "wait_unit", lambda *_: None)
    monkeypatch.setattr(
        watch, "digest", lambda path: "scene" if path.name == "scene.xml" else "audit"
    )
    monkeypatch.setattr(watch, "report", lambda *_args, **_kwargs: ({}, {}, source_hash))
    calls = []
    monkeypatch.setattr(watch.subprocess, "run", lambda command, **kwargs: calls.append(command))
    return banks, calls


def test_only_audited_banks_start_training(monkeypatch, tmp_path):
    _, calls = configure(monkeypatch, tmp_path)
    watch.main()
    assert len(calls) == 1
    assert calls[0][1:3] == ["-m", "scripts.rsi_train_causal_return_detector"]
    assert calls[0].count("--original-source") == 3


def test_missing_audit_fails_closed(monkeypatch, tmp_path):
    banks, calls = configure(monkeypatch, tmp_path)
    (banks[1].parent / "audit" / "complete.json").unlink()
    with pytest.raises(ValueError, match="independent audit"):
        watch.main()
    assert not calls


def test_bad_source_hash_fails_closed(monkeypatch, tmp_path):
    _, calls = configure(monkeypatch, tmp_path)
    monkeypatch.setattr(watch, "report", lambda *_args, **_kwargs: ({}, {}, "wrong"))
    with pytest.raises(ValueError, match="source and scene"):
        watch.main()
    assert not calls
