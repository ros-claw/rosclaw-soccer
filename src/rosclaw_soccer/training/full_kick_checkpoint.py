"""Bounded numeric checkpoint decoding for the explicit full554 kick schema.

No pickle, executable model, policy activation or hardware authority. A content
hash binds bytes, not behavioral qualification or permission to use a policy.
"""

from __future__ import annotations

import io
import re
import zipfile
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from rosclaw_soccer.sim.contracts import hash_bytes

MAXIMUM_CHECKPOINT_BYTES = 4 * 1024 * 1024


def full_kick_checkpoint_shapes() -> dict[str, tuple[int, ...]]:
    """Exact full actor, contextual projection, critic and exploration state."""
    shapes: dict[str, tuple[int, ...]] = {
        "context_adapter.weight": (512, 6),
        "logstd": (29,),
    }
    for head, widths in (("actor", (547, 512, 256, 128, 29)), ("critic", (553, 256, 128, 1))):
        for index, (n_in, n_out) in enumerate(zip(widths[:-1], widths[1:], strict=True)):
            shapes[f"{head}.{2 * index}.weight"] = (n_out, n_in)
            shapes[f"{head}.{2 * index}.bias"] = (n_out,)
    return shapes


def load_full_kick_checkpoint(
    path: Path, *, expected_hash: str
) -> tuple[dict[str, NDArray[np.float32]], str]:
    """Validate ZIP members and NPY headers before allocating declared arrays."""
    if (
        not isinstance(path, Path)
        or not isinstance(expected_hash, str)
        or re.fullmatch(r"sha256:[0-9a-f]{64}", expected_hash) is None
    ):
        raise ValueError("explicit checkpoint path and SHA256 required")
    # Bound the actual read, not just a racy preceding stat(). All subsequent
    # decoding consumes these same hash-verified bytes, not a reopened file.
    with path.open("rb") as stream:
        raw = stream.read(MAXIMUM_CHECKPOINT_BYTES + 1)
    if len(raw) > MAXIMUM_CHECKPOINT_BYTES:
        raise ValueError("full-kick checkpoint exceeds byte bound")
    digest = str(hash_bytes(raw))
    if digest != expected_hash:
        raise ValueError("full-kick checkpoint hash mismatch")
    shapes = full_kick_checkpoint_shapes()
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        entries = archive.infolist()
        if (
            len(entries) != len(shapes)
            or {entry.filename for entry in entries} != {key + ".npy" for key in shapes}
            or sum(entry.file_size for entry in entries) > MAXIMUM_CHECKPOINT_BYTES
        ):
            raise ValueError("exact bounded full-kick ZIP members required")
        for key, shape in shapes.items():
            entry = archive.getinfo(key + ".npy")
            with archive.open(entry) as stream:
                version = np.lib.format.read_magic(stream)  # type: ignore[no-untyped-call]
                if version == (1, 0):
                    declared, _, dtype = np.lib.format.read_array_header_1_0(stream)  # type: ignore[no-untyped-call]
                elif version == (2, 0):
                    declared, _, dtype = np.lib.format.read_array_header_2_0(stream)  # type: ignore[no-untyped-call]
                else:
                    raise ValueError("unsupported full-kick NPY version")
                if (
                    declared != shape
                    or dtype != np.dtype(np.float32)
                    or entry.file_size != stream.tell() + int(np.prod(shape)) * 4
                ):
                    raise ValueError("bounded float32 full-kick tensor header required")
    with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
        parameters = {key: archive[key].copy() for key in shapes}
    for value in parameters.values():
        if not np.isfinite(value).all() or (np.abs(value) > 1e4).any():
            raise ValueError("finite bounded full-kick parameters required")
        value.setflags(write=False)
    return parameters, digest
