from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from rosclaw_soccer.evidence.age04 import load_age04_manifest, validate_age04_manifest


def _hash(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _write_case(root: Path, name: str, payload: dict[str, Any]) -> tuple[str, str, str, str]:
    case_root = root / name
    case_root.mkdir(parents=True)
    evidence = case_root / "evidence.json"
    media = case_root / "media.mp4"
    evidence.write_text(json.dumps(payload), encoding="utf-8")
    media.write_bytes((name + "-video").encode())
    return (
        f"{name}/evidence.json",
        _hash(evidence),
        f"{name}/media.mp4",
        _hash(media),
    )


def _manifest(tmp_path: Path) -> tuple[Path, Path]:
    evidence_root = tmp_path / "external"
    body_hash = "sha256:" + "a" * 64
    passed = {
        "activation_ceiling": "SIM_ONLY",
        "body_hash": body_hash,
        "evidence_domain": "SIM",
        "passed": True,
        "strict_replay": True,
        "result": {"target_error_m": 0.04},
    }
    development = {
        "activation_ceiling": "SIM_ONLY",
        "body_hash": body_hash,
        "evidence_domain": "DEVELOPMENT_SHOWCASE",
        "passed": False,
        "strict_replay": True,
        "result": {"actuator_saturation_steps": 7},
    }
    pass_paths = _write_case(evidence_root, "pass-case", passed)
    dev_paths = _write_case(evidence_root, "dev-case", development)
    value = {
        "schema_version": "rosclaw_soccer.age04_manifest.v1",
        "player_id": "claw7",
        "academy_age": 4,
        "activation_ceiling": "SIM_ONLY",
        "source_repository": "ros-claw/rosclaw",
        "source_commit": "b" * 40,
        "cases": [
            {
                "case_id": "pass-case",
                "title": "Certified",
                "verdict": "PASS",
                "media_label": "CERTIFIED",
                "evidence_relpath": pass_paths[0],
                "evidence_sha256": pass_paths[1],
                "media_relpath": pass_paths[2],
                "media_sha256": pass_paths[3],
                "body_hash": body_hash,
                "facts": {"passed": True, "result.target_error_m": 0.04},
            },
            {
                "case_id": "dev-case",
                "title": "Development",
                "verdict": "DEVELOPMENT",
                "media_label": "DEVELOPMENT",
                "evidence_relpath": dev_paths[0],
                "evidence_sha256": dev_paths[1],
                "media_relpath": dev_paths[2],
                "media_sha256": dev_paths[3],
                "body_hash": body_hash,
                "facts": {"passed": False, "result.actuator_saturation_steps": 7},
            },
        ],
        "reel": [
            {
                "case_id": "pass-case",
                "chapter": "Certified chapter",
                "subtitle": "Strict replay",
                "start_sec": 0.0,
                "duration_sec": 2.0,
            },
            {
                "case_id": "dev-case",
                "chapter": "Development chapter",
                "subtitle": "Clearly labelled",
                "start_sec": 0.0,
                "duration_sec": 2.0,
            },
        ],
    }
    manifest = tmp_path / "project" / "evidence" / "manifests" / "age04.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps(value), encoding="utf-8")
    return manifest, evidence_root


def test_validates_certified_and_development_boundaries(tmp_path: Path) -> None:
    manifest_path, evidence_root = _manifest(tmp_path)
    report = validate_age04_manifest(manifest_path, evidence_root)
    assert report.passed
    assert [case.verdict for case in report.cases] == ["PASS", "DEVELOPMENT"]
    assert sum(case.facts_verified for case in report.cases) == 4


def test_media_tamper_fails_closed(tmp_path: Path) -> None:
    manifest_path, evidence_root = _manifest(tmp_path)
    (evidence_root / "pass-case" / "media.mp4").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="media hash mismatch"):
        validate_age04_manifest(manifest_path, evidence_root)


def test_development_media_cannot_hide_its_label(tmp_path: Path) -> None:
    manifest_path, _ = _manifest(tmp_path)
    value = json.loads(manifest_path.read_text(encoding="utf-8"))
    value["cases"][1]["media_label"] = "CERTIFIED"
    manifest_path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="exact public verdict label"):
        load_age04_manifest(manifest_path)


def test_relative_paths_cannot_escape_evidence_root(tmp_path: Path) -> None:
    manifest_path, _ = _manifest(tmp_path)
    value = json.loads(manifest_path.read_text(encoding="utf-8"))
    value["cases"][0]["media_relpath"] = "../outside.mp4"
    manifest_path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="normalized relative POSIX"):
        load_age04_manifest(manifest_path)
