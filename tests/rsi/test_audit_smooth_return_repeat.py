import numpy as np
import pytest

from scripts.rsi_audit_smooth_return_repeat import compare_arrays


def test_repeat_checks_keys_shapes_dtypes_and_values():
    first = {"qpos": np.array([[1.0, 2.0]])}
    compare_arrays(first, {"qpos": first["qpos"].copy()})
    for bad in (
        {},
        {"qvel": first["qpos"]},
        {"qpos": first["qpos"].astype(np.float32)},
        {"qpos": np.array([[1.0, 3.0]])},
        {"qpos": np.array([1.0, 2.0])},
    ):
        with pytest.raises(ValueError):
            compare_arrays(first, bad)
