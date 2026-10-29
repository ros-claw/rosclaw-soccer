import hashlib
import json
from types import SimpleNamespace

import pytest

from scripts import rsi_chain_neural_relay_audit as chain


def test_unit_parser_and_rejection(monkeypatch):
    monkeypatch.setattr(
        chain.subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(
            stdout="LoadState=loaded\nActiveState=inactive\nResult=success\nExecMainStatus=0\n"
        ),
    )
    assert chain.producer_status("rosclaw-good.service") == ("inactive", "success", 0)
    with pytest.raises(ValueError):
        chain.producer_status("other.service")


def test_receipt_hash_binds_exact_content_and_contract(tmp_path):
    receipt = {
        "source_head": "a" * 40,
        "status": "TEST",
        "hardware_authorized": False,
        "promotion_authorized": False,
        "goal_complete": False,
    }
    path = tmp_path / "complete.json"
    path.write_text(json.dumps(receipt))
    assert (
        chain.receipt_hash(tmp_path, "a" * 40, "TEST")
        == "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
    )
    with pytest.raises(ValueError):
        chain.receipt_hash(tmp_path, "b" * 40, "TEST")
    receipt["hardware_authorized"] = True
    path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError):
        chain.receipt_hash(tmp_path, "a" * 40, "TEST")


def test_main_passes_computed_digest_as_one_literal_argument(monkeypatch, tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    path = source / "complete.json"
    path.write_text(
        json.dumps(
            {
                "source_head": "a" * 40,
                "status": "TEST",
                "hardware_authorized": False,
                "promotion_authorized": False,
                "goal_complete": False,
            }
        )
    )
    calls = []
    monkeypatch.setattr(chain, "clean", lambda head: None)
    monkeypatch.setattr(chain, "producer_status", lambda unit: ("inactive", "success", 0))
    monkeypatch.setattr(
        chain.subprocess, "run", lambda args, **kwargs: calls.append((args, kwargs))
    )
    monkeypatch.setattr(
        chain.sys,
        "argv",
        [
            "chain",
            "--unit",
            "rosclaw-test.service",
            "--source",
            str(source),
            "--output",
            str(tmp_path / "audit"),
            "--source-head",
            "a" * 40,
            "--source-status",
            "TEST",
            "--expected-head",
            "a" * 40,
        ],
    )
    chain.main()
    args, kwargs = calls[0]
    assert (
        args[args.index("--source-hash") + 1]
        == "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
    )
    assert kwargs == {"check": True}
