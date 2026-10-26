import numpy as np
import pytest
from test_candidate_body23_comparison import _models

from rosclaw_soccer.rsi.reference_support_ik_audit import audit_support_candidate
from rosclaw_soccer.rsi.shared_target_forecast import compiled_model_hash


def test_independent_distance_readback_never_integrates():
    model, _ = _models()
    source = np.array([0, 0, 1, 1, 0, 0, 0], dtype=np.float64)
    target = source.copy()
    target[2] += 0.01
    joints = np.zeros(29, dtype=np.float64)
    result = audit_support_candidate(
        model,
        source,
        joints,
        target,
        joints.copy(),
        np.array([True, True]),
        expected_model_hash=compiled_model_hash(model),
    )
    np.testing.assert_allclose(
        np.array(result["candidate_foot_gaps_m"]) - result["source_foot_gaps_m"], 0.01, atol=1e-12
    )
    assert result["integrated_native_steps"] == 0
    assert result["physical_execution_qualified"] is False


def test_audit_rejects_unbounded_or_unrelated_joint_change():
    model, _ = _models()
    root = np.array([0, 0, 1, 1, 0, 0, 0], dtype=np.float64)
    joints = np.zeros(29, dtype=np.float64)
    target = joints.copy()
    target[0] = 0.21
    with pytest.raises(ValueError, match="exceeds"):
        audit_support_candidate(
            model,
            root,
            joints,
            root.copy(),
            target,
            np.array([True, True]),
            expected_model_hash=compiled_model_hash(model),
        )
    target = joints.copy()
    target[20] = 0.01
    with pytest.raises(ValueError, match="unchanged"):
        audit_support_candidate(
            model,
            root,
            joints,
            root.copy(),
            target,
            np.array([True, True]),
            expected_model_hash=compiled_model_hash(model),
        )
