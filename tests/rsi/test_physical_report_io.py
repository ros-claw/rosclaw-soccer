"""Codec fixtures do not count as new physical executions."""

import gzip
import json
from pathlib import Path

import pytest

from rosclaw_soccer.rsi.failure_curriculum_evidence import _sealed
from rosclaw_soccer.rsi.physical_report_io import load_physical_report, resolve_physical_report
from rosclaw_soccer.sim.contracts import hash_json
from scripts.rsi_atomic_artifacts import write_once


def sample() -> dict:
    value = {
        "schema": "fixture.physical_report.v1",
        "actions": [[0.0, -0.0, 0.0000000123456789, -987.6543210987654]] * 100,
        "hardware_authorized": False,
    }
    return {**value, "report_hash": hash_json(value)}


@pytest.mark.parametrize("compressed", [False, True])
def test_full_payload_and_seal_roundtrip(tmp_path: Path, compressed: bool) -> None:
    path = tmp_path / ("report.json.gz" if compressed else "report.json")
    value = sample()
    write_once(path, value)
    assert resolve_physical_report(tmp_path / "report.json") == path
    assert load_physical_report(tmp_path / "report.json") == value
    assert _sealed(tmp_path / "report.json") == value
    assert load_physical_report(path) == value


def test_ambiguous_representations_rejected_even_identical(tmp_path: Path) -> None:
    for name in ("report.json", "report.json.gz"):
        write_once(tmp_path / name, sample())
    for name in ("report.json", "report.json.gz"):
        with pytest.raises(ValueError, match="exactly one"):
            load_physical_report(tmp_path / name)


def test_incomplete_directory_is_not_complete_evidence(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="exactly one"):
        resolve_physical_report(tmp_path / "report.json")
    (tmp_path / "report.json.gz").mkdir()
    with pytest.raises(ValueError, match="exactly one"):
        resolve_physical_report(tmp_path / "report.json")


def test_corrupt_gzip_and_changed_numerical_seal_rejected(tmp_path: Path) -> None:
    path = tmp_path / "report.json.gz"
    path.write_bytes(b"not gzip")
    with pytest.raises(gzip.BadGzipFile):
        load_physical_report(path)
    value = sample()
    value["actions"][0][2] += 1e-14
    with gzip.open(path, "wt", encoding="utf-8") as stream:
        json.dump(value, stream)
    with pytest.raises(ValueError, match="unsealed"):
        _sealed(path)


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_gzip_rejected(tmp_path: Path, invalid: float) -> None:
    path = tmp_path / "report.json.gz"
    with gzip.open(path, "wt", encoding="utf-8") as stream:
        json.dump({"measurement": invalid}, stream)
    with pytest.raises(ValueError):
        load_physical_report(path)


def test_unrelated_artifact_name_not_resolved_as_physics(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="filename"):
        resolve_physical_report(tmp_path / "validation_summary.json")
