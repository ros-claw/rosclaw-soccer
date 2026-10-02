"""Lossless local sampling views sharing one complete, immutable mean model.

This is persistence only, not a new distribution, policy, or authorization.
Ordinary historical JSON/gzip views are returned unchanged.
"""

import re
from pathlib import Path
from typing import Any, cast

from rosclaw_soccer.rsi.json_artifact_io import load_json_artifact


def load_sampling_model(path: Path) -> dict[str, Any]:
    value = load_json_artifact(path)
    if value.get("schema") != "rosclaw.growth.shared_json_payload.v1":
        return value
    from rosclaw.growth.shared_proof_payload import restore_payload

    digest = value.get("payload_hash")
    if (
        path.parent.name != "models"
        or value.get("location") != ["mean_model"]
        or type(digest) is not str
        or not re.fullmatch(r"sha256:[0-9a-f]{64}", digest)
    ):
        raise ValueError("exact local mean-model sampling envelope required")
    store = path.parent.parent / ".shared-models"
    payload = store / f"{digest[7:]}.json.gz"
    if (
        path.is_symlink()
        or path.parent.is_symlink()
        or path.parent.parent.is_symlink()
        or store.is_symlink()
        or payload.is_symlink()
        or not payload.is_file()
    ):
        raise ValueError("complete local non-symlink sampling mean required")
    result = cast(dict[str, Any], restore_payload(value, load_json_artifact(payload)))
    if result.get("schema") not in (
        "soccer.rsi.smooth_memory_sampling.v1",
        "soccer.rsi.output_memory_step_sampling.v1",
    ):
        raise ValueError("declared memory sampling view required")
    return result
