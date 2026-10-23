import hashlib

import numpy as np
import pytest

from scripts.rsi_qualify_student_pass_continuations import compare_trajectory


def artifacts(root, *, changed_after=80, mismatch=False):
    root.mkdir()
    out = {}
    for key in ("initial", "physical", "A", "B"):
        a = np.zeros(100, np.float32)
        if key != "initial":
            a[changed_after:] = 1 if mismatch else 0
        path = root / (key + ".npz")
        np.savez_compressed(path, value=a)
        out[key] = dict(
            path=path.name, hash="sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
        )
    return dict(artifacts=out)


def test_prefix_requires_actual_bit_identity_not_whole_suffix(tmp_path):
    old = artifacts(tmp_path / "old")
    new = artifacts(tmp_path / "new", mismatch=True)
    assert compare_trajectory(tmp_path / "new", new, tmp_path / "old", old, prefix=80) == 4
    with pytest.raises(ValueError, match="identity mismatch"):
        compare_trajectory(tmp_path / "new", new, tmp_path / "old", old, prefix=None)


def test_one_wrong_early_neural_or_native_value_rejects(tmp_path):
    old = artifacts(tmp_path / "old")
    new = artifacts(tmp_path / "new", changed_after=79, mismatch=True)
    with pytest.raises(ValueError, match="identity mismatch"):
        compare_trajectory(tmp_path / "new", new, tmp_path / "old", old, prefix=80)


def test_saved_source_tamper_rejects(tmp_path):
    old = artifacts(tmp_path / "old")
    new = artifacts(tmp_path / "new")
    np.savez_compressed(tmp_path / "old/A.npz", value=np.ones(100, np.float32))
    with pytest.raises(ValueError, match="original whole"):
        compare_trajectory(tmp_path / "new", new, tmp_path / "old", old, prefix=80)
