"""Pinned, license-aware catalog for recovery-foundation research baselines."""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from rosclaw_soccer.sim.contracts import hash_json


class ReferenceUse(StrEnum):
    RECOVERY_BASELINE = "RECOVERY_BASELINE"
    MULTI_POSTURE_BASELINE = "MULTI_POSTURE_BASELINE"
    CURRICULUM_REFERENCE = "CURRICULUM_REFERENCE"
    ADVERSARIAL_MOTION_PRIOR_REFERENCE = "ADVERSARIAL_MOTION_PRIOR_REFERENCE"


@dataclass(frozen=True)
class RecoveryReferenceSpec:
    name: str
    repository_url: str
    expected_commit: str
    license_id: str
    local_directory: str
    entrypoint: str
    body: str
    simulator_backend: str
    used_as: ReferenceUse
    paper_arxiv_id: str
    code_reuse_allowed: bool
    limitations: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if (
            not self.name
            or len(self.expected_commit) != 40
            or not all(character in "0123456789abcdef" for character in self.expected_commit)
            or not self.repository_url.startswith("https://github.com/")
            or Path(self.local_directory).is_absolute()
            or Path(self.entrypoint).is_absolute()
            or not self.paper_arxiv_id.replace(".", "").isdigit()
        ):
            raise ValueError("recovery reference specification is invalid")
        if self.license_id == "UNDECLARED" and self.code_reuse_allowed:
            raise ValueError("unlicensed recovery reference cannot allow code reuse")

    @property
    def spec_hash(self) -> str:
        payload = asdict(self)
        payload["used_as"] = self.used_as.value
        return str(hash_json(payload))


PINNED_RECOVERY_REFERENCES = (
    RecoveryReferenceSpec(
        name="HumanUP",
        repository_url="https://github.com/RunpeiDong/humanup",
        expected_commit="7516e0f27e6f4d1e7365cf64ea577a78247bd8cb",
        license_id="Apache-2.0",
        local_directory="HumanUP",
        entrypoint="simulation/legged_gym/legged_gym/scripts/train.py",
        body="Unitree G1",
        simulator_backend="Isaac Gym",
        used_as=ReferenceUse.RECOVERY_BASELINE,
        paper_arxiv_id="2502.12152",
        code_reuse_allowed=True,
        limitations=("Supine and prone policies are separately trained.",),
    ),
    RecoveryReferenceSpec(
        name="HoST",
        repository_url="https://github.com/InternRobotics/HoST",
        expected_commit="70bb580949a336a920833700e4b5dc3bf7fe87ce",
        license_id="MIT",
        local_directory="HoST",
        entrypoint="legged_gym/legged_gym/scripts/train.py",
        body="Unitree G1",
        simulator_backend="Isaac Gym",
        used_as=ReferenceUse.MULTI_POSTURE_BASELINE,
        paper_arxiv_id="2502.08378",
        code_reuse_allowed=True,
        limitations=(
            "Joint supine-and-prone G1 training remains an upstream TODO.",
            "The upstream G1 prone policy only claims incidental side-lying handling.",
        ),
    ),
    RecoveryReferenceSpec(
        name="HiFAR",
        repository_url="https://github.com/taohuang13/hifar",
        expected_commit="5a5cef76eab33fdc1a6ae46cad5447e9f9d83ad0",
        license_id="Apache-2.0",
        local_directory="HiFAR",
        entrypoint="train.py",
        body="Booster T1",
        simulator_backend="Isaac Gym",
        used_as=ReferenceUse.CURRICULUM_REFERENCE,
        paper_arxiv_id="2502.20061",
        code_reuse_allowed=True,
        limitations=("Body and actuator contracts require explicit G1 adaptation.",),
    ),
    RecoveryReferenceSpec(
        name="AMP_mjlab",
        repository_url="https://github.com/ccrpRepo/AMP_mjlab",
        expected_commit="6c7a2947fccc973e4af8e6d90e550400f1b6fcfc",
        license_id="UNDECLARED",
        local_directory="AMP_mjlab",
        entrypoint="scripts/train.py",
        body="Unitree G1",
        simulator_backend="MuJoCo Playground",
        used_as=ReferenceUse.ADVERSARIAL_MOTION_PRIOR_REFERENCE,
        paper_arxiv_id="2605.18611",
        code_reuse_allowed=False,
        limitations=(
            "No repository license was present at the pinned commit.",
            "Read-only architectural reference; no source may be copied or merged.",
        ),
    ),
)


def _head_commit(repository: Path) -> str | None:
    if not (repository / ".git").exists():
        return None
    result = subprocess.run(
        ("git", "-C", str(repository), "rev-parse", "HEAD"),
        check=False,
        capture_output=True,
        text=True,
        timeout=5.0,
    )
    if result.returncode != 0:
        return None
    value = result.stdout.strip().lower()
    return value if len(value) == 40 else None


def audit_recovery_reference_catalog(repository_root: Path) -> dict[str, Any]:
    """Audit local clones without executing any third-party training code."""

    root = repository_root.expanduser().resolve()
    rows = []
    all_qualified = True
    reusable_count = 0
    for reference in PINNED_RECOVERY_REFERENCES:
        repository = root / reference.local_directory
        head = _head_commit(repository)
        entrypoint_exists = (repository / reference.entrypoint).is_file()
        license_files = sorted(path.name for path in repository.glob("LICENSE*") if path.is_file())
        commit_matches = head == reference.expected_commit
        license_boundary_ok = bool(license_files) == (reference.license_id != "UNDECLARED")
        qualified = bool(
            repository.is_dir() and commit_matches and entrypoint_exists and license_boundary_ok
        )
        reusable = bool(qualified and reference.code_reuse_allowed)
        all_qualified &= qualified
        reusable_count += int(reusable)
        row = {
            "name": reference.name,
            "repository_url": reference.repository_url,
            "expected_commit": reference.expected_commit,
            "observed_commit": head,
            "commit_matches": commit_matches,
            "license_id": reference.license_id,
            "license_files": license_files,
            "license_boundary_ok": license_boundary_ok,
            "entrypoint": reference.entrypoint,
            "entrypoint_exists": entrypoint_exists,
            "body": reference.body,
            "simulator_backend": reference.simulator_backend,
            "used_as": reference.used_as.value,
            "paper_arxiv_id": reference.paper_arxiv_id,
            "code_reuse_allowed": reference.code_reuse_allowed,
            "qualified": qualified,
            "reusable": reusable,
            "limitations": list(reference.limitations),
            "spec_hash": reference.spec_hash,
        }
        row["row_hash"] = hash_json(row)
        rows.append(row)
    report: dict[str, Any] = {
        "schema_version": "rosclaw_soccer.recovery_reference_audit.v1",
        "reference_count": len(rows),
        "all_repositories_qualified": all_qualified,
        "code_reusable_reference_count": reusable_count,
        "execution_boundary": "SOURCE_AND_LICENSE_AUDIT_ONLY",
        "activation_ceiling": "SIM_ONLY",
        "hardware_authorized": False,
        "references": rows,
    }
    report["report_hash"] = hash_json(report)
    return report


def write_recovery_reference_audit(*, repository_root: Path, output_path: Path) -> dict[str, Any]:
    report = audit_recovery_reference_catalog(repository_root)
    destination = output_path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
    return report


__all__ = [
    "PINNED_RECOVERY_REFERENCES",
    "RecoveryReferenceSpec",
    "ReferenceUse",
    "audit_recovery_reference_catalog",
    "write_recovery_reference_audit",
]
