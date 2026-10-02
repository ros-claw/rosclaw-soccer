"""Lossless physical report transport, never an evidence or policy promotion.

Only report.json/report.json.gz are alternative representations. Reject an
ambiguous directory even if its two documents happen to have identical hashes.
Legacy reports are neither rewritten nor removed by this reader.
"""

from pathlib import Path
from typing import Any

from rosclaw_soccer.rsi.json_artifact_io import load_json_artifact


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
    return load_json_artifact(resolve_physical_report(path))
