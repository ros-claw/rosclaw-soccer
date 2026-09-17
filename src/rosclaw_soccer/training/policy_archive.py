"""Preserve explicitly selected football assets without granting composition authority."""

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from rosclaw_soccer.sim.contracts import hash_json


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def preserve_policy_assets(
    entries: list[dict[str, Any]], *, output: Path, maximum_bytes: int
) -> dict[str, Any]:
    """New immutable-by-convention snapshot with independent, verified file copies.

    Existence and matching hashes do not qualify a skill. Evidence files are
    preserved as artifacts, not interpreted as promotion. No checkpoint loading,
    executable deserialization, motor registration or policy activation occurs.
    """
    if not entries or type(maximum_bytes) is not int or maximum_bytes <= 0:
        raise ValueError("explicit asset list and positive archive budget required")
    rows = []
    unique: dict[str, Path] = {}
    names: set[str] = set()
    for entry in entries:
        name = entry.get("name")
        roles = entry.get("roles")
        if (
            not isinstance(name, str)
            or re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,79}", name) is None
            or name in names
            or not isinstance(roles, list)
            or not roles
            or any(
                r not in ("all", "playmaker", "finisher", "defender", "goalkeeper") for r in roles
            )
            or not isinstance(entry.get("scope"), str)
            or not entry["scope"].strip()
        ):
            raise ValueError("unique asset name, explicit role and scope required")
        path = Path(entry["path"]).resolve(strict=True)
        if not path.is_file():
            raise ValueError("regular asset file required")
        digest = file_digest(path)
        if entry.get("expected_hash", digest) != digest:
            raise ValueError("source asset hash mismatch")
        names.add(name)
        unique[digest] = path
        rows.append(
            dict(
                name=name,
                roles=roles.copy(),
                scope=entry["scope"],
                original_path=str(path),
                file_hash=digest,
                archive_path="objects/" + digest[7:],
            )
        )
    total = sum(p.stat().st_size for p in unique.values())
    if total > maximum_bytes:
        raise ValueError("archive exceeds declared byte budget")
    output.mkdir(parents=True, exist_ok=False)
    (output / "objects").mkdir()
    written = 0
    for digest, source in unique.items():
        destination = output / "objects" / digest[7:]
        copied = 0
        with source.open("rb") as src, destination.open("xb") as dst:
            for block in iter(lambda: src.read(1024 * 1024), b""):
                copied += len(block)
                written += len(block)
                if copied > source.stat().st_size or written > maximum_bytes:
                    raise ValueError("source changed or grew during archive")
                dst.write(block)
        if file_digest(destination) != digest:
            raise ValueError("copied asset hash mismatch; incomplete snapshot retained")
    report = dict(
        schema="soccer.policy_preservation.v1",
        entries=rows,
        unique_bytes=total,
        unique_objects=len(unique),
        composition_qualified=False,
        activation_ceiling="SIM_ONLY",
        promoted=False,
    )
    report["manifest_hash"] = hash_json(report)
    with (output / "manifest.json").open("x") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    return report
