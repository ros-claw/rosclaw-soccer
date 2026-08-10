"""Content-bound Academy Age-4 evidence and media-source validation."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import Any

_MAX_MANIFEST_BYTES = 512 * 1024
_MAX_EVIDENCE_BYTES = 16 * 1024 * 1024
_SHA256_PREFIX = "sha256:"


class Verdict(StrEnum):
    """Public status of one physics-backed media case."""

    PASS = "PASS"
    DEVELOPMENT = "DEVELOPMENT"
    REJECTED = "REJECTED"


@dataclass(frozen=True)
class EvidenceCase:
    case_id: str
    title: str
    verdict: Verdict
    media_label: str
    evidence_relpath: str
    evidence_sha256: str
    media_relpath: str
    media_sha256: str
    body_hash: str
    facts: tuple[tuple[str, Any], ...]

    def __post_init__(self) -> None:
        if not self.case_id or not self.title:
            raise ValueError("Age-4 evidence case identity is empty")
        _validate_relative_path(self.evidence_relpath)
        _validate_relative_path(self.media_relpath)
        _validate_sha256(self.evidence_sha256, "evidence hash")
        _validate_sha256(self.media_sha256, "media hash")
        _validate_sha256(self.body_hash, "body hash")
        if self.verdict is not Verdict.PASS and self.media_label != self.verdict.value:
            raise ValueError("non-passing media must carry its exact public verdict label")
        if self.verdict is Verdict.PASS and self.media_label not in {"CERTIFIED", "PASS"}:
            raise ValueError("passing Age-4 media must be labelled CERTIFIED or PASS")
        if not self.facts:
            raise ValueError("evidence case must bind at least one measured fact")


@dataclass(frozen=True)
class ReelSegment:
    case_id: str
    chapter: str
    subtitle: str
    start_sec: float
    duration_sec: float

    def __post_init__(self) -> None:
        if not self.case_id or not self.chapter or not self.subtitle:
            raise ValueError("Age-4 reel segment labels must be non-empty")
        if not math.isfinite(self.start_sec) or self.start_sec < 0.0:
            raise ValueError("Age-4 reel segment start must be finite and non-negative")
        if not math.isfinite(self.duration_sec) or not 1.0 <= self.duration_sec <= 90.0:
            raise ValueError("Age-4 reel segment duration must be in [1, 90] seconds")


@dataclass(frozen=True)
class Age04Manifest:
    player_id: str
    academy_age: int
    activation_ceiling: str
    source_repository: str
    source_commit: str
    cases: tuple[EvidenceCase, ...]
    reel: tuple[ReelSegment, ...]
    schema_version: str = "rosclaw_soccer.age04_manifest.v1"

    def __post_init__(self) -> None:
        if self.player_id != "claw7" or self.academy_age != 4:
            raise ValueError("this manifest must bind the Claw-7 Academy Age-4 baseline")
        if self.activation_ceiling != "SIM_ONLY":
            raise ValueError("Age-4 public evidence must remain SIM_ONLY")
        if self.source_repository != "ros-claw/rosclaw":
            raise ValueError("Age-4 source repository identity is invalid")
        if len(self.source_commit) != 40 or any(
            character not in "0123456789abcdef" for character in self.source_commit
        ):
            raise ValueError("Age-4 source commit must be a lowercase Git SHA-1")
        if self.schema_version != "rosclaw_soccer.age04_manifest.v1":
            raise ValueError("unsupported Age-4 manifest schema")
        case_ids = tuple(case.case_id for case in self.cases)
        if not case_ids or len(case_ids) != len(set(case_ids)):
            raise ValueError("Age-4 case ids must be non-empty and unique")
        if not self.reel or any(segment.case_id not in case_ids for segment in self.reel):
            raise ValueError("every Age-4 reel segment must reference a known evidence case")
        if not any(case.verdict is Verdict.DEVELOPMENT for case in self.cases):
            raise ValueError("Age-4 manifest must preserve the development boundary")


@dataclass(frozen=True)
class CaseValidation:
    case_id: str
    verdict: str
    evidence_path: str
    media_path: str
    strict_replay: bool
    passed_claim: bool
    facts_verified: int


@dataclass(frozen=True)
class ValidationReport:
    player_id: str
    academy_age: int
    activation_ceiling: str
    manifest_hash: str
    cases: tuple[CaseValidation, ...]
    passed: bool
    schema_version: str = "rosclaw_soccer.age04_validation.v1"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_age04_manifest(path: Path) -> Age04Manifest:
    """Load the bounded committed manifest without accessing external evidence."""

    resolved = path.expanduser().resolve()
    if not resolved.is_file() or resolved.stat().st_size > _MAX_MANIFEST_BYTES:
        raise ValueError("Age-4 manifest is missing or exceeds the size limit")
    value = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Age-4 manifest must be a JSON object")
    raw_cases = value.pop("cases", None)
    raw_reel = value.pop("reel", None)
    if not isinstance(raw_cases, list) or not isinstance(raw_reel, list):
        raise ValueError("Age-4 manifest cases and reel must be arrays")
    cases: list[EvidenceCase] = []
    for raw_case in raw_cases:
        if not isinstance(raw_case, dict):
            raise ValueError("Age-4 case must be an object")
        facts = raw_case.pop("facts", None)
        if not isinstance(facts, dict):
            raise ValueError("Age-4 case facts must be an object")
        raw_case["verdict"] = Verdict(raw_case["verdict"])
        raw_case["facts"] = tuple(sorted(facts.items()))
        cases.append(EvidenceCase(**raw_case))
    reel = tuple(ReelSegment(**segment) for segment in raw_reel)
    return Age04Manifest(cases=tuple(cases), reel=reel, **value)


def validate_age04_manifest(manifest_path: Path, evidence_root: Path) -> ValidationReport:
    """Verify hashes, physics verdicts, body identity, and measured facts fail-closed."""

    manifest_file = manifest_path.expanduser().resolve()
    manifest = load_age04_manifest(manifest_file)
    root = evidence_root.expanduser().resolve()
    if not root.is_dir():
        raise ValueError("ROSClaw Soccer evidence root does not exist")
    validations: list[CaseValidation] = []
    for case in manifest.cases:
        evidence = _external_file(root, case.evidence_relpath)
        media = _external_file(root, case.media_relpath)
        if evidence.stat().st_size > _MAX_EVIDENCE_BYTES:
            raise ValueError(f"{case.case_id} evidence exceeds the size limit")
        if _hash_file(evidence) != case.evidence_sha256:
            raise ValueError(f"{case.case_id} evidence hash mismatch")
        if _hash_file(media) != case.media_sha256:
            raise ValueError(f"{case.case_id} media hash mismatch")
        payload = json.loads(evidence.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"{case.case_id} evidence must be a JSON object")
        if payload.get("body_hash") != case.body_hash:
            raise ValueError(f"{case.case_id} body hash mismatch")
        if payload.get("activation_ceiling") != "SIM_ONLY":
            raise ValueError(f"{case.case_id} evidence is not SIM_ONLY")
        strict_replay = _strict_replay(payload)
        passed_claim = payload.get("passed") is True
        if not strict_replay:
            raise ValueError(f"{case.case_id} does not provide strict replay evidence")
        if case.verdict is Verdict.PASS:
            if not passed_claim or "DEVELOPMENT" in str(payload.get("evidence_domain", "")):
                raise ValueError(f"{case.case_id} cannot enter the certified gallery")
        elif case.verdict is Verdict.DEVELOPMENT:
            if "DEVELOPMENT" not in str(payload.get("evidence_domain", "")):
                raise ValueError(f"{case.case_id} lacks a development evidence domain")
        elif passed_claim:
            raise ValueError(f"{case.case_id} rejected media contradicts a passing claim")
        for fact_path, expected in case.facts:
            actual = _lookup(payload, fact_path)
            if not _equivalent(actual, expected):
                raise ValueError(
                    f"{case.case_id} fact mismatch at {fact_path}: {actual!r} != {expected!r}"
                )
        validations.append(
            CaseValidation(
                case_id=case.case_id,
                verdict=case.verdict.value,
                evidence_path=str(evidence),
                media_path=str(media),
                strict_replay=strict_replay,
                passed_claim=passed_claim,
                facts_verified=len(case.facts),
            )
        )
    return ValidationReport(
        player_id=manifest.player_id,
        academy_age=manifest.academy_age,
        activation_ceiling=manifest.activation_ceiling,
        manifest_hash=_hash_file(manifest_file),
        cases=tuple(validations),
        passed=True,
    )


def _validate_relative_path(value: str) -> None:
    path = PurePosixPath(value)
    if not value or path.is_absolute() or ".." in path.parts or "\\" in value:
        raise ValueError("evidence paths must be normalized relative POSIX paths")


def _validate_sha256(value: str, label: str) -> None:
    if not value.startswith(_SHA256_PREFIX):
        raise ValueError(f"{label} must use the sha256: prefix")
    digest = value.removeprefix(_SHA256_PREFIX)
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError(f"{label} must be a lowercase sha256 digest")


def _external_file(root: Path, relative: str) -> Path:
    candidate = (root / relative).resolve()
    if candidate == root or root not in candidate.parents or not candidate.is_file():
        raise ValueError(f"external evidence path is missing or escapes its root: {relative}")
    return candidate


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return _SHA256_PREFIX + digest.hexdigest()


def _strict_replay(payload: dict[str, Any]) -> bool:
    if payload.get("strict_replay") is True:
        return True
    cases = payload.get("cases")
    return bool(
        isinstance(cases, list)
        and cases
        and all(isinstance(case, dict) and case.get("strict_replay") is True for case in cases)
    )


def _lookup(value: Any, dotted_path: str) -> Any:
    current = value
    for token in dotted_path.split("."):
        if isinstance(current, dict) and token in current:
            current = current[token]
        elif isinstance(current, list) and token.isdigit() and int(token) < len(current):
            current = current[int(token)]
        else:
            raise ValueError(f"evidence fact path does not exist: {dotted_path}")
    return current


def _equivalent(actual: Any, expected: Any) -> bool:
    if isinstance(actual, bool) or isinstance(expected, bool):
        return actual is expected
    if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
        return math.isfinite(float(actual)) and math.isclose(
            float(actual), float(expected), rel_tol=1e-9, abs_tol=1e-9
        )
    return bool(actual == expected)


__all__ = [
    "Age04Manifest",
    "CaseValidation",
    "EvidenceCase",
    "ReelSegment",
    "ValidationReport",
    "Verdict",
    "load_age04_manifest",
    "validate_age04_manifest",
]
