"""Bind an already-declared read-only audit after its producer exits.

This permits a specific audit successor, not model selection, experiment
generation, retry, promotion or an LLM wakeup mechanism.
"""

import hashlib
import json
import re
from pathlib import Path


def bind_complete(
    root: Path,
    *,
    complete_hash: str | None,
    producer_head: str | None,
    declaration_hash: str | None,
) -> str:
    def digest(path: Path) -> str:
        if path.is_symlink() or not path.is_file() or path.stat().st_size > 64 * 1024 * 1024:
            raise ValueError("bounded ordinary completed evidence required")
        return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()

    cp, dp = root / "complete.json", root / "declaration.json"
    if root.is_symlink() or (root / "failed.json").exists():
        raise ValueError("failed or substituted producer cannot start an audit")
    if complete_hash is not None:
        if producer_head is not None or declaration_hash is not None or digest(cp) != complete_hash:
            raise ValueError("one unambiguous exact complete commitment required")
        return complete_hash
    if (
        producer_head is None
        or re.fullmatch(r"[0-9a-f]{40}", producer_head) is None
        or declaration_hash is None
        or digest(dp) != declaration_hash
    ):
        raise ValueError(
            "queued successor requires preregistered declaration and full producer head"
        )
    actual = digest(cp)
    complete, declaration = [json.loads(p.read_text()) for p in (cp, dp)]
    if (
        complete.get("source_head") != producer_head
        or declaration.get("source_head") != producer_head
        or complete.get("declaration_hash") != declaration_hash
        or any(
            p.get(k) is not False
            for p in (complete, declaration)
            for k in ("hardware_authorized", "promotion_authorized")
        )
    ):
        raise ValueError("completed receipt must match exact simulation-only preregistration")
    return actual
