"""Publish complete JSON artifacts atomically, without replacing existing data.

POSIX local-filesystem experiment utility. A waiter must never observe the
destination before the whole document has been written and fsynced.
"""

import gzip
import io
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from rosclaw_soccer.rsi.json_artifact_io import load_json_artifact
from rosclaw_soccer.sim.contracts import hash_json


def write_once(path: Path, value: dict[str, Any]) -> None:
    expected = hash_json(value)  # Reject nonfinite/nonserializable input first.

    def check_existing() -> None:
        if hash_json(load_json_artifact(path)) != expected:
            raise ValueError(f"resume commitment differs: {path}")

    if path.exists():
        check_existing()
        return
    descriptor, name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    temporary = Path(name)
    try:
        if path.name.endswith(".json.gz"):
            with os.fdopen(descriptor, "wb") as binary:
                with gzip.GzipFile(filename="", fileobj=binary, mode="wb", mtime=0) as compressed:
                    text = io.TextIOWrapper(compressed, encoding="utf-8")
                    try:
                        json.dump(value, text, indent=2, sort_keys=True, allow_nan=False)
                        text.write("\n")
                        text.flush()
                    finally:
                        text.detach()
                binary.flush()
                os.fsync(binary.fileno())
        else:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
        try:
            # Unlike replace/rename-overwrite, link fails if another writer has
            # already published. Both writers must agree on the whole payload.
            os.link(temporary, path)
        except FileExistsError:
            check_existing()
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        # Only this invocation's exact private temporary path is removed.
        temporary.unlink(missing_ok=True)
