"""Admit one frozen, independently audited successful balance teacher."""

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


def balance_teacher_proof(
    root: Path,
    commitment: str,
    audit_root: Path,
    audit_commitment: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, str], np.ndarray]:
    def read(path: Path, h: str) -> dict[str, Any]:
        if path.is_symlink() or (path.parent / "failed.json").exists():
            raise ValueError("ordinary closed balance proof required")
        raw = path.read_bytes()
        if "sha256:" + hashlib.sha256(raw).hexdigest() != h:
            raise ValueError("balance proof commitment changed")
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise ValueError("balance proof must be a JSON object")
        return value

    if root.is_symlink() or audit_root.is_symlink():
        raise ValueError("ordinary closed balance roots required")
    rp, ap = root / "complete.json", audit_root / "complete.json"
    r, a = read(rp, commitment), read(ap, audit_commitment)
    pins = dict(a["input_file_hashes"])
    dp = root / "declaration.json"
    d = read(dp, pins[str(dp)])
    latent = np.array(r["selected_latent"], dtype=np.float64)
    if (
        r["status"] != "PAID_ACTIVE_BALANCE_TEACHER_PILOT_CLOSED_PENDING_AUDIT"
        or a["status"] != "PAID_ACTIVE_BALANCE_ALL_PRIMITIVE_AND_CONTINUATION_AUDIT_PASS"
        or pins.get(str(rp)) != commitment
        or d.get("recovery_start") != "reference-end"
        or r.get("recovery_start") != "reference-end"
        or r["selected_rank"][:3] != [2, 2, 100]
        or d["pilot_cases"] != [0, 15]
        or len(d["source_cases"]) != 16
        or d["max_delta_rad"] != 0.2
        or d["max_delta_per_control_rad"] != 0.02
        or d["new_phase_controls"] != 100
        or r["neural_optimizer_updates"] != 0
        or latent.shape != (9, 4)
        or not np.isfinite(latent).all()
        or np.abs(latent).max() > 1
        or a["counts"]["producer_native"] != r["actual_native_steps"]
        or a["counts"]["producer_neural"] != r["actual_neural_rows"]
        or a["own_native_steps"] != 0
    ):
        raise ValueError("exact audited reference-end feedback model required")
    neural = 0
    if len(r["final"]) != 2:
        raise ValueError("both declared pilot conditions required")
    for i, replicas in zip((0, 15), r["final"], strict=True):
        if len(replicas) != 2:
            raise ValueError("both full-world repeats required")
        for world in replicas:
            rec, third = world["recovery"], world["third"]
            if (
                world["case"] != i
                or rec["healthy"] is not True
                or rec["recovery_stability_observed"] is not True
                or third is None
                or third["body_failed"] is not False
                or world["continuation"]["status"] != "READY"
            ):
                raise ValueError("actual stable recovery and subsequent motion required")
            neural += third["neural_rows"]
    if a["own_neural_rows"] != neural:
        raise ValueError("all subsequent own neural actions must be independently audited")
    pins.update({str(rp): commitment, str(ap): audit_commitment})
    return r, d, pins, latent
