"""Verify or install pinned, external research references without importing them.

The manifest is JSON syntax (also valid YAML 1.2) to keep this bootstrap on the
standard library. Source and model-weight licenses are recorded separately.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ("git", "-C", str(root), *args),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _validate_entry(entry: object) -> dict[str, Any]:
    if not isinstance(entry, dict):
        raise ValueError("reference entry must be an object")
    required = {
        "name",
        "repository",
        "commit",
        "license_file",
        "license_sha256",
        "license_id",
        "weight_license",
        "purpose",
        "reused_as",
        "files_reviewed",
        "copied_code",
    }
    if set(entry) != required or not all(
        isinstance(entry[key], str) for key in required - {"files_reviewed", "copied_code"}
    ):
        raise ValueError("reference entry has missing, extra, or non-string fields")
    value = dict(entry)
    if (
        value["copied_code"] is not False
        or not isinstance(value["files_reviewed"], list)
        or not value["files_reviewed"]
        or any(not isinstance(p, str) for p in value["files_reviewed"])
    ):
        raise ValueError("reference must declare reviewed files and no copied code")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", value["name"]):
        raise ValueError("reference name must be a safe directory component")
    if not re.fullmatch(
        r"https://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+(?:\.git)?", value["repository"]
    ):
        raise ValueError("reference must use a direct GitHub repository URL")
    if not re.fullmatch(r"[0-9a-f]{40}", value["commit"]):
        raise ValueError("reference requires a full pinned commit")
    if not re.fullmatch(r"[0-9a-f]{64}", value["license_sha256"]):
        raise ValueError("reference requires the license text SHA-256")
    license_path = Path(value["license_file"])
    if license_path.is_absolute() or ".." in license_path.parts or not license_path.parts:
        raise ValueError("license file must remain inside its checkout")
    if not value["purpose"] or not value["reused_as"]:
        raise ValueError("reference purpose and reuse type must be declared")
    return value


def inspect_reference(root: Path, entry: dict[str, Any], *, install: bool) -> dict[str, str]:
    target = root / entry["name"]
    if not target.exists():
        if not install:
            return {"name": entry["name"], "status": "MISSING"}
        root.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            (
                "git",
                "clone",
                "--filter=blob:none",
                "--no-checkout",
                entry["repository"],
                str(target),
            ),
            check=True,
        )
        _git(target, "checkout", "--detach", entry["commit"])
    if not (target / ".git").exists():
        raise ValueError(f"{entry['name']}: target is not a Git checkout")
    remote = _git(target, "remote", "get-url", "origin")
    if remote.removeprefix("https://ghfast.top/").removesuffix(".git") != entry[
        "repository"
    ].removesuffix(".git"):
        raise ValueError(f"{entry['name']}: origin URL differs from manifest")
    head = _git(target, "rev-parse", "HEAD")
    if head != entry["commit"]:
        raise ValueError(f"{entry['name']}: checkout SHA differs from manifest")
    if _git(target, "status", "--porcelain"):
        raise ValueError(f"{entry['name']}: reference checkout has local changes")
    license_path = target / entry["license_file"]
    if not license_path.is_file() or license_path.is_symlink():
        raise ValueError(f"{entry['name']}: license text missing or linked")
    actual = hashlib.sha256(license_path.read_bytes()).hexdigest()
    if actual != entry["license_sha256"]:
        raise ValueError(f"{entry['name']}: license text hash differs from manifest")
    license_text = license_path.read_text(encoding="utf-8")
    marker = {"MIT": "MIT License", "Apache-2.0": "Apache License"}.get(entry["license_id"])
    if marker is None or marker not in license_text:
        raise ValueError(f"{entry['name']}: declared source license marker missing")
    return {"name": entry["name"], "status": "VERIFIED", "commit": head, "license_sha256": actual}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "references" / "REFERENCE_MANIFEST.yaml",
    )
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="External checkout root; never the Soccer source tree",
    )
    parser.add_argument(
        "--install", action="store_true", help="Clone only missing pinned references"
    )
    args = parser.parse_args()
    source = Path(__file__).resolve().parents[1]
    root = args.root.expanduser().resolve()
    if root == source or source in root.parents or root in source.parents:
        raise ValueError("reference checkout root must be outside the Soccer repository")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if (
        not isinstance(manifest, dict)
        or set(manifest) != {"schema", "references", "pending"}
        or manifest["schema"] != "rosclaw_soccer.reference_manifest.v1"
    ):
        raise ValueError("unknown reference manifest schema")
    entries = [_validate_entry(item) for item in manifest["references"]]
    if len({item["name"] for item in entries}) != len(entries):
        raise ValueError("reference names must be unique")
    if not isinstance(manifest["pending"], list):
        raise ValueError("pending references must be a list")
    rows = [inspect_reference(root, item, install=args.install) for item in entries]
    print(
        json.dumps(
            {"schema": manifest["schema"], "results": rows, "pending": manifest["pending"]},
            sort_keys=True,
            indent=2,
        )
    )
    if any(row["status"] != "VERIFIED" for row in rows):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
