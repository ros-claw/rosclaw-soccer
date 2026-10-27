import numpy as np
import pytest

from scripts.rsi_audit_relay_fall_predictor import predict


def test_numpy_replay_and_shape_fail_closed(tmp_path):
    path = tmp_path / "model.npz"
    np.savez(
        path,
        center=np.zeros(104, np.float32),
        scale=np.ones(104, np.float32),
        weight1=np.zeros((64, 104), np.float32),
        bias1=np.zeros(64, np.float32),
        weight2=np.zeros((1, 64), np.float32),
        bias2=np.array([1], np.float32),
    )
    assert predict(path, np.zeros((2, 104), np.float32)).tolist() == [True, True]
    with pytest.raises(ValueError):
        predict(path, np.zeros((2, 103), np.float32))
