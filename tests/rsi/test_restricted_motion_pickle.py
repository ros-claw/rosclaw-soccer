import hashlib
import os
import pathlib
import pickle

import numpy as np
import pytest

from rosclaw_soccer.rsi.restricted_motion_pickle import read_motion


def _data():
    return {
        pathlib.Path("../motion.npz"): {
            "root_trans_offset": np.zeros((2, 3), np.float32),
            "root_rot": np.zeros((2, 4), np.float32),
            "dof": np.zeros((2, 23), np.float32),
            "pose_aa": np.zeros((2, 27, 3), np.float32),
            "contact_mask": np.zeros((2, 2), np.float64),
            "fps": np.array(30, np.int64),
        }
    }


def _write(path, data):
    blob = pickle.dumps(data, protocol=5)
    path.write_bytes(blob)
    return "sha256:" + hashlib.sha256(blob).hexdigest()


def test_read_pin_and_validate_without_qualifying_motion(tmp_path):
    path = tmp_path / "motion.pkl"
    result = read_motion(path, expected_sha256=_write(path, _data()))
    assert result["fields"]["dof"].shape == (2, 23)
    assert int(result["fields"]["fps"]) == 30
    assert result["motion_name"] == "motion.npz"
    assert result["joint_order_verified"] is False
    assert result["physical_execution_qualified"] is False


def test_unapproved_global_rejected_without_calling_it(tmp_path):
    class Evil:
        def __reduce__(self):
            return os.system, ("exit 42",)

    path = tmp_path / "evil.pkl"
    digest = _write(path, Evil())
    with pytest.raises(pickle.UnpicklingError, match="unapproved"):
        read_motion(path, expected_sha256=digest)


def test_hash_and_shape_must_match(tmp_path):
    path = tmp_path / "motion.pkl"
    data = _data()
    next(iter(data.values()))["dof"] = np.zeros((2, 29), np.float32)
    digest = _write(path, data)
    with pytest.raises(ValueError):
        read_motion(path, expected_sha256="sha256:" + "0" * 64)
    with pytest.raises(ValueError):
        read_motion(path, expected_sha256=digest)


def test_no_trailing_pickle_stream(tmp_path):
    path = tmp_path / "motion.pkl"
    blob = pickle.dumps(_data(), protocol=5) + b"garbage"
    path.write_bytes(blob)
    with pytest.raises(ValueError, match="trailing"):
        read_motion(path, expected_sha256="sha256:" + hashlib.sha256(blob).hexdigest())


def test_symlink_is_rejected_even_when_target_digest_matches(tmp_path):
    path = tmp_path / "motion.pkl"
    digest = _write(path, _data())
    link = tmp_path / "link.pkl"
    link.symlink_to(path)
    with pytest.raises(ValueError):
        read_motion(link, expected_sha256=digest)


def test_nonfinite_values_cannot_become_a_reference(tmp_path):
    path = tmp_path / "motion.pkl"
    data = _data()
    next(iter(data.values()))["dof"][0, 0] = np.nan
    digest = _write(path, data)
    with pytest.raises(ValueError, match="finite"):
        read_motion(path, expected_sha256=digest)
