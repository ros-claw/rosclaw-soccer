"""Metadata/CLI rejection tests; no physical qualification."""

import json
import sys
from dataclasses import asdict

import pytest

from rosclaw_soccer.training.expanded_entry_courses import expanded_entry_courses
from rosclaw_soccer.training.receiving_course_preflight import preflight_receiving_courses
from scripts import rsi_expanded_bilateral_review_common as reader
from scripts import rsi_review_expanded_bilateral as cli


def test_active_producer_is_not_read_or_restarted(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "subprocess.check_output",
        lambda *a, **kw: "MainPID=123\nSubState=start\nResult=success\nExecMainStatus=0\n",
    )
    with pytest.raises(ValueError, match="actually be closed"):
        reader.admit_closed("unit", tmp_path / "missing", "head")
    assert not (tmp_path / "missing").exists()


def bank_fixture(tmp_path, monkeypatch):
    from scripts.rsi_native_context_recovery_common import digest

    monkeypatch.setattr(
        "subprocess.check_output",
        lambda *a, **kw: "MainPID=0\nSubState=exited\nResult=success\nExecMainStatus=0\n",
    )
    bank = expanded_entry_courses()
    h = preflight_receiving_courses(bank)
    d = dict(
        schema="soccer.rsi.bilateral_expanded40.v1",
        source_head="head",
        shard=0,
        bank_hash=h,
        courses=[asdict(c) | dict(course=list(c.course)) for c in bank],
        input_file_hashes={},
    )
    rows = []
    for item in bank:
        if item.course_index % 4:
            continue
        folder = tmp_path / f"course{item.course_index:02d}"
        folder.mkdir()
        p = folder / "fixture.bin"
        p.write_bytes(b"not-a-native-recording")
        receipt = dict(artifact=p.name, artifact_hash=digest(p))
        row = dict(
            course=asdict(item) | dict(course=list(item.course)),
            candidates=[receipt] * 9,
            baseline=receipt,
            selected=receipt,
            shared_endpoint=True,
            forecast_native_steps=25200,
            endpoint_native_steps=3000,
            neural_optimizer_updates=0,
        )
        (folder / "complete.json").write_text(json.dumps(row))
        rows.append(row)
    c = dict(
        source_head="head",
        shard=0,
        bank_hash=h,
        results=rows,
        forecast_native_steps=252000,
        endpoint_native_steps=30000,
        neural_optimizer_updates=0,
    )
    (tmp_path / "declaration.json").write_text(json.dumps(d))
    (tmp_path / "complete.json").write_text(json.dumps(c))
    return c


def test_metadata_acceptance_is_only_a_read_hash_contract_not_physics(tmp_path, monkeypatch):
    bank_fixture(tmp_path, monkeypatch)
    _, result, pins = reader.admit_closed("unit", tmp_path, "head")
    assert len(result["results"]) == 10 and str(tmp_path / "complete.json") in pins
    # The deliberately non-NPZ payload could NEVER pass the native reviewer.


@pytest.mark.parametrize("kind", ["head", "missing", "steps", "artifact"])
def test_wrong_closed_producer_contract_is_fatal(tmp_path, monkeypatch, kind):
    c = bank_fixture(tmp_path, monkeypatch)
    if kind == "head":
        c["source_head"] = "other"
    elif kind == "missing":
        c["results"].pop()
    elif kind == "steps":
        c["endpoint_native_steps"] += 2800
    else:
        (tmp_path / "course00/fixture.bin").write_bytes(b"changed")
    (tmp_path / "complete.json").write_text(json.dumps(c))
    with pytest.raises(ValueError):
        reader.admit_closed("unit", tmp_path, "head")


def test_wrong_cli_head_before_any_output_or_models(tmp_path, monkeypatch):
    args = [
        "review",
        "--mode",
        "native",
        "--producer",
        "unit",
        str(tmp_path / "producer"),
        "--expected-head",
        "expected",
        "--expected-producer-head",
        "producer-head",
    ]
    for name in ("world", "motion-csv", "core-source", "output"):
        args += ["--" + name, str(tmp_path / name)]
    monkeypatch.setattr(sys, "argv", args)
    monkeypatch.setattr(cli.subprocess, "check_output", lambda *a, **kw: "wrong\n")
    with pytest.raises(ValueError, match="sealed source"):
        cli.main()
    assert not (tmp_path / "output").exists()
