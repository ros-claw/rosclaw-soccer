"""Read a physically AND neurally audited continuing recovery reference."""

import hashlib
import json
from pathlib import Path
from typing import Any


def recovery_proof(
    root: Path,
    commitment: str,
    audit_root: Path,
    audit_commitment: str,
    *,
    cases: list[dict[str, Any]],
    course_bank: str = "train",
) -> tuple[dict[str, Any], dict[str, str]]:
    def read(path: Path, expected: str) -> dict[str, Any]:
        if path.is_symlink():
            raise ValueError("ordinary bound recovery proof required")
        raw = path.read_bytes()
        if "sha256:" + hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError("recovery evidence hash changed")
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise ValueError("ordinary JSON object recovery proof required")
        return value

    if course_bank not in ("train", "dev"):
        raise ValueError("explicit train/dev source identity required")
    rp, ap = root / "complete.json", audit_root / "complete.json"
    if any(p.is_symlink() or (p / "failed.json").exists() for p in (root, audit_root)):
        raise ValueError("ordinary closed recovery roots without failures required")
    r, audit = read(rp, commitment), read(ap, audit_commitment)
    if (
        r["status"] != "PAID_CONTINUING_CHAIN_CPU_EXAM_CLOSED_PENDING_INDEPENDENT_AUDIT"
        or audit["status"] != "PAID_CONTINUING_CHAIN_NATIVE_AND_NEURAL_AUDIT_PASS"
        or audit["input_file_hashes"].get(str(rp)) != commitment
        or audit["own_native_steps"] != 0
        or audit["own_neural_rows"] != r["actual_neural_rows"]
        or r["skills"] != 3
        or not r["all_independent_chain_repeats_exact"]
        or r.get("recovery_teacher") is not None
    ):
        raise ValueError("completed pure-neural three-skill recovery audit required")
    dp = root / "declaration.json"
    d = read(dp, audit["input_file_hashes"][str(dp)])
    if (
        d["cases"] != cases
        or d["body_resets_between_skills"] != 0
        or d.get("course_bank", "train") != course_bank
        or not d["fixed_world_goal"]
    ):
        raise ValueError("same declared physical courses and fixed-body protocol required")
    pins = dict(audit["input_file_hashes"])
    pins.update({str(rp): commitment, str(ap): audit_commitment})
    return d, pins


def unique_proof_path(pins: dict[str, str], commitment: str) -> Path:
    paths = [Path(p) for p, h in pins.items() if h == commitment]
    if len(paths) != 1:
        raise ValueError("one unambiguous bound actor artifact required")
    return paths[0]
