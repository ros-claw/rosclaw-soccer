import numpy as np

from scripts.rsi_audit_neural_receiver_torque import effort_limits


def test_auditor_effort_matches_hg29_physical_contract():
    limits = effort_limits()
    assert limits.shape == (29,) and np.isfinite(limits).all()
    assert limits[:6].tolist() == [88, 139, 88, 139, 50, 50]
    assert limits[6:12].tolist() == [88, 139, 88, 139, 50, 50]
    assert limits[12:15].tolist() == [88, 50, 50]
    assert limits[15:22].tolist() == [25, 25, 25, 25, 25, 5, 5]
