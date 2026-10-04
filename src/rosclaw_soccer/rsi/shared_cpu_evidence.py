"""Opt-in, lossless storage of new simulation snapshots; no evidence migration."""

import os
import tempfile
from pathlib import Path
from typing import Any

from rosclaw.growth.shared_blob_store import publish_readonly_blob

from rosclaw_soccer.sim.contracts import hash_bytes


def save_shared_compiled_model(model: Any, target: Path) -> str:
    """Save a fresh native model, then publish a read-only same-byte hardlink.

    Historical files are never converted, deleted or overwritten. MuJoCo is
    imported only for this simulation-asset serialization, not robot execution.
    The generic Core store performs hash/budget/no-overwrite validation.
    """
    import mujoco

    if (
        target.name != "compiled_model.mjb"
        or not target.parent.is_dir()
        or target.exists()
        or target.is_symlink()
        or any(p.is_symlink() for p in (target.parent, *target.parent.parents))
    ):
        raise ValueError("new local compiled simulation snapshot required")
    if not 1 <= int(mujoco.mj_sizeModel(model)) <= 256 * 1024**2:
        raise ValueError("compiled simulation snapshot exceeds declared serialization budget")
    store = target.parent.parent / ".shared-worlds"
    if store.is_symlink():
        raise ValueError("local shared world store required")
    store.mkdir(mode=0o700, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=".serialized-world-", dir=store)
    os.close(descriptor)
    temporary = Path(name)
    try:
        mujoco.mj_saveModel(model, str(temporary), None)
        expected = hash_bytes(temporary.read_bytes())
        record = publish_readonly_blob(
            temporary,
            store=store,
            target=target,
            expected_hash=expected,
            maximum_blob_bytes=256 * 1024**2,
        )
        return str(record.blob_hash)
    finally:
        temporary.unlink(missing_ok=True)
