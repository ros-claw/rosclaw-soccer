"""Bind the active imported Core source before simulator/learner allocation."""

import importlib.util
import subprocess
from pathlib import Path


def pin_active_core(root: Path, expected_head: str, pins: dict) -> Path:
    from scripts.rsi_native_context_recovery_common import digest

    resolved = root.resolve()
    head = subprocess.check_output(
        ["git", "-C", str(resolved), "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(resolved), "status", "--porcelain"], text=True
    )
    spec = importlib.util.find_spec("rosclaw")
    package = resolved / "src/rosclaw"
    if (
        head != expected_head
        or dirty
        or spec is None
        or spec.origin is None
        or Path(spec.origin).resolve().parent != package
    ):
        raise ValueError("clean exact Core commit must match active import resolution")
    files = sorted(package.rglob("*.py"))
    if not files or any(path.is_symlink() for path in files):
        raise ValueError("ordinary Core source files required")
    for path in files:
        value = digest(path)
        if str(path) in pins and pins[str(path)] != value:
            raise ValueError("active Core source conflicts with predecessor commitment")
        pins[str(path)] = value
    return resolved
