"""Admission of three independently replayed, unpromoted balance teachers."""

import hashlib
import json
from pathlib import Path

import numpy as np


def _digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


def admit_relay_balance_teachers(pairs: list[list[str]]) -> tuple[list[np.ndarray], dict[str, str]]:
    if type(pairs) is not list or len(pairs) != 3 or any(len(p) != 4 for p in pairs):
        raise ValueError("all three explicit source/replay/hash pairs required")
    latents, pins = [], {}
    for source_name, audit_name, source_hash, audit_hash in pairs:
        source, audit = Path(source_name), Path(audit_name)
        if (
            any(p.is_symlink() or not p.is_file() for p in (source, audit))
            or _digest(source) != source_hash
            or _digest(audit) != audit_hash
        ):
            raise ValueError("exact immutable balance source and audit required")
        if str(source) in pins or str(audit) in pins:
            raise ValueError("distinct teacher provenance required")
        s, a = (json.loads(p.read_text()) for p in (source, audit))
        if (
            s.get("status") != "PAID_ACTIVE_BALANCE_TEACHER_PILOT_CLOSED_PENDING_AUDIT"
            or a.get("status") != "PAID_ACTIVE_BALANCE_ALL_PRIMITIVE_AND_CONTINUATION_AUDIT_PASS"
            or a.get("input_file_hashes", {}).get(str(source)) != source_hash
            or s.get("all_selected_recovery_and_third_repeats_exact") is not True
            or any(
                row.get(key) is not False
                for row in (s, a)
                for key in ("goal_complete", "hardware_authorized", "promotion_authorized")
            )
        ):
            raise ValueError("unpromoted source linked to its own native replay required")
        latent = np.asarray(s["selected_latent"], dtype=np.float64)
        if latent.shape != (9, 4) or not np.isfinite(latent).all() or (np.abs(latent) > 1).any():
            raise ValueError("bounded actual feedback teacher matrix required")
        latents.append(latent)
        pins[str(source)], pins[str(audit)] = source_hash, audit_hash
    return latents, pins
