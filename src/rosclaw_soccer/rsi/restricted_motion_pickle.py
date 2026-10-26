"""Narrow research-data reader, not a robot command or arbitrary pickle loader.

Use only for integrity-pinned files in a resource-limited research worker.
Pickle is an executable format; this restricted reader is defense in depth,
not a sandbox or a substitute for trustworthy provenance.
"""

import hashlib
import importlib
import io
import pathlib
import pickle
import re
from typing import Any

import numpy as np

_MAX_FILE_BYTES = 8 * 1024 * 1024


class _RestrictedReader(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> Any:
        if (module, name) == ("pathlib", "PosixPath"):
            return pathlib.PosixPath
        if (module, name) == ("numpy", "dtype"):
            return np.dtype
        if (module, name) in (
            ("numpy.core.numeric", "_frombuffer"),
            ("numpy._core.numeric", "_frombuffer"),
        ):
            numpy_core = (
                "numpy._core.numeric"
                if int(np.__version__.split(".")[0]) >= 2
                else "numpy.core.numeric"
            )
            return importlib.import_module(numpy_core)._frombuffer
        raise pickle.UnpicklingError("unapproved pickle global")

    def persistent_load(self, pid: Any) -> Any:
        raise pickle.UnpicklingError("persistent pickle objects forbidden")


def read_motion(path: pathlib.Path, *, expected_sha256: str) -> dict[str, Any]:
    """Return one validated, detached 23-DoF reference; mapping remains unknown."""
    if (
        type(path) is not pathlib.PosixPath
        or path.is_symlink()
        or not path.is_file()
        or type(expected_sha256) is not str
        or re.fullmatch(r"sha256:[0-9a-f]{64}", expected_sha256) is None
        or path.stat().st_size > _MAX_FILE_BYTES
    ):
        raise ValueError("bounded non-symlink file and explicit digest required")
    blob = path.read_bytes()
    if (
        len(blob) > _MAX_FILE_BYTES
        or "sha256:" + hashlib.sha256(blob).hexdigest() != expected_sha256
    ):
        raise ValueError("exact bounded input digest required")
    stream = io.BytesIO(blob)
    value = _RestrictedReader(stream).load()
    if stream.read(1) or type(value) is not dict or len(value) != 1:
        raise ValueError("one motion and no trailing pickle stream required")
    key, motion = next(iter(value.items()))
    fields = {"root_trans_offset", "root_rot", "dof", "fps", "contact_mask", "pose_aa"}
    if (
        type(key) not in (str, pathlib.PosixPath)
        or type(motion) is not dict
        or set(motion) != fields
    ):
        raise ValueError("one explicitly structured reference required")
    arrays = {name: motion[name] for name in fields}
    if any(type(a) is not np.ndarray or a.dtype.kind not in "fiu" for a in arrays.values()):
        raise ValueError("numeric owned arrays only; no object arrays")
    n = len(arrays["dof"]) if arrays["dof"].ndim else 0
    shapes = {
        "root_trans_offset": (n, 3),
        "root_rot": (n, 4),
        "dof": (n, 23),
        "contact_mask": (n, 2),
        "pose_aa": (n, 27, 3),
        "fps": (),
    }
    if (
        not 2 <= n <= 60000
        or any(arrays[name].shape != shape for name, shape in shapes.items())
        or any(not np.isfinite(a).all() for a in arrays.values())
        or arrays["fps"].dtype.kind not in "iu"
        or not 1 <= int(arrays["fps"]) <= 240
        or any(a.nbytes > 32 * 1024 * 1024 for a in arrays.values())
    ):
        raise ValueError("finite bounded reference arrays and explicit integer fps required")
    return dict(
        motion_name=pathlib.PurePath(key).name,
        source_hash=expected_sha256,
        fields={name: value.copy() for name, value in arrays.items()},
        joint_order_verified=False,
        body_compatibility_verified=False,
        physical_execution_qualified=False,
        hardware_authorized=False,
    )
