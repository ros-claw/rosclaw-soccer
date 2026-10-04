"""Lossless physical report transport, never an evidence or policy promotion.

Only report.json/report.json.gz are alternative representations. Reject an
ambiguous directory even if its two documents happen to have identical hashes.
Legacy reports are neither rewritten nor removed by this reader.
"""

import re
from pathlib import Path
from typing import Any, cast

from rosclaw_soccer.rsi.json_artifact_io import load_json_artifact

MODEL_LOCATION = ("contact_motor_policy", "step_motor_proof", "model")
MEAN_MODEL_LOCATION = (*MODEL_LOCATION, "mean_model")
CPU_MODEL_LOCATION = ("executed_motor_policy", "step_motor_proof", "model")
CPU_MEAN_MODEL_LOCATION = (*CPU_MODEL_LOCATION, "mean_model")
SHARED_SCHEMA = "rosclaw.growth.shared_json_payload.v1"


def resolve_physical_report(path: Path) -> Path:
    if path.name not in ("report.json", "report.json.gz"):
        raise ValueError("physical report filename required")
    alternatives = (path.parent / "report.json", path.parent / "report.json.gz")
    present = [candidate for candidate in alternatives if candidate.exists()]
    if len(present) != 1 or not present[0].is_file():
        raise ValueError(f"exactly one complete physical report required: {path.parent}")
    return present[0]


def load_physical_report(path: Path) -> dict[str, Any]:
    """Read the complete numerical payload; caller still verifies its seals."""
    resolved = resolve_physical_report(path)
    value = load_json_artifact(resolved)
    if value.get("schema") != SHARED_SCHEMA:
        return value
    from rosclaw.growth.shared_proof_payload import restore_payload

    digest = value.get("payload_hash")
    if (
        value.get("location")
        not in (
            list(MODEL_LOCATION),
            list(MEAN_MODEL_LOCATION),
            list(CPU_MODEL_LOCATION),
            list(CPU_MEAN_MODEL_LOCATION),
        )
        or type(digest) is not str
        or not re.fullmatch(r"sha256:[0-9a-f]{64}", digest)
    ):
        raise ValueError("exact content-addressed physical model proof required")
    store = resolved.parent.parent / ".shared-models"
    payload_path = store / f"{digest[7:]}.json.gz"
    if (
        resolved.is_symlink()
        or resolved.parent.is_symlink()
        or store.is_symlink()
        or payload_path.is_symlink()
        or not payload_path.is_file()
    ):
        raise ValueError("complete local non-symlink shared model proof required")
    payload = load_json_artifact(payload_path)
    return cast(dict[str, Any], restore_payload(value, payload))
