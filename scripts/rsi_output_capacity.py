"""Refuse a new physical trial before known output-space exhaustion."""

import errno
import shutil
from pathlib import Path


def require_output_capacity(path: Path, minimum_free_bytes: int) -> int:
    if type(minimum_free_bytes) is not int or minimum_free_bytes <= 0:
        raise ValueError("positive explicit output-space reserve required")
    parent = path
    while not parent.exists():
        parent = parent.parent
    free = shutil.disk_usage(parent).free
    if free < minimum_free_bytes:
        raise OSError(errno.ENOSPC, "insufficient evidence space BEFORE new physics", str(path))
    return free
