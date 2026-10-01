"""Lossless JSON/gzip-JSON input; no policy, execution or activation semantics."""

import gzip
import json
from pathlib import Path
from typing import Any

from rosclaw_soccer.sim.contracts import hash_json


def load_json_artifact(path: Path) -> dict[str, Any]:
    if path.name.endswith(".json.gz"):
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            value = json.load(stream)
    else:
        value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("mapping JSON artifact required")
    hash_json(value)
    return value
