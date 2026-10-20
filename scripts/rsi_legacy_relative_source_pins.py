"""Bind legacy relative FK source pins to their already-pinned frozen bootstrap.

Never rewrites the original evidence or interprets an old relative path in the
new experiment's checkout. Only the legacy src/*.py surface is accepted.
"""

import hashlib
import subprocess
from pathlib import Path


def resolve_source_pins(pins, root, expected_head):
    relative = [p for p in pins if not Path(p).is_absolute()]
    if not relative:
        return dict(pins), {}
    if root is None or expected_head is None or not root.is_absolute() or root.is_symlink():
        raise ValueError("legacy relative source pins require an explicit frozen source root")
    if (
        subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
        != expected_head
    ):
        raise ValueError("legacy source HEAD differs")
    bootstrap = root / "scripts/rsi_audit_sagittal_reference.py"
    if pins.get(str(bootstrap)) != "sha256:" + hashlib.sha256(bootstrap.read_bytes()).hexdigest():
        raise ValueError("legacy source root must already be identified by its pinned FK bootstrap")
    resolved, mapping = {p: h for p, h in pins.items() if p not in relative}, {}
    for name in relative:
        path = Path(name)
        if (
            path.parts[0] != "src"
            or ".." in path.parts
            or "." in path.parts
            or path.suffix != ".py"
        ):
            raise ValueError("only legacy relative src Python pins can be rebound")
        absolute = root / path
        if not absolute.resolve().is_relative_to(root.resolve()) or absolute.is_symlink():
            raise ValueError("legacy source pin escapes frozen source root")
        expected = pins[name]
        if "sha256:" + hashlib.sha256(absolute.read_bytes()).hexdigest() != expected:
            raise ValueError("legacy source bytes differ; never substitute current source")
        if str(absolute) in resolved and resolved[str(absolute)] != expected:
            raise ValueError("legacy absolute source commitment conflicts")
        resolved[str(absolute)] = expected
        mapping[name] = str(absolute)
    return resolved, mapping
