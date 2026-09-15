"""Explicit zero-extension migration, not a learned or promoted receiving skill."""

from collections.abc import Mapping

import numpy as np


def migrate_receiving_heading(parameters: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    shapes: dict[str, tuple[int, ...]] = {"logstd": (29,)}
    for name, output in (("actor", 29), ("critic", 1)):
        shapes.update(
            {
                f"{name}.0.weight": (128, 133),
                f"{name}.0.bias": (128,),
                f"{name}.2.weight": (128, 128),
                f"{name}.2.bias": (128,),
                f"{name}.4.weight": (output, 128),
                f"{name}.4.bias": (output,),
            }
        )
    if set(parameters) != set(shapes) or any(
        not isinstance(parameters[k], np.ndarray)
        or parameters[k].shape != shape
        or parameters[k].dtype != np.float32
        or not np.isfinite(parameters[k]).all()
        or np.max(np.abs(parameters[k])) > 1e6
        for k, shape in shapes.items()
    ):
        raise ValueError("finite explicit 133-feature receiving state dictionary required")
    result = {k: v.copy() for k, v in parameters.items()}
    for name in ("actor", "critic"):
        key = name + ".0.weight"
        result[key] = np.concatenate((result[key], np.zeros((128, 2), dtype=np.float32)), axis=1)
    return result
