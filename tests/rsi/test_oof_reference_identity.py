import numpy as np
import pytest

from rosclaw_soccer.rsi.oof_reference_identity import exact_reference_arrays
from scripts.rsi_audit_oof_reference_identity import study_predictions


def test_reference_study_selection_does_not_confuse_in_sample_with_oof():
    risk = {"ball_augmented": {"predictions": [0]}, "in_sample_rehearsal": {"predictions": [1]}}
    oof = {
        "risk_variant": "ball_augmented",
        "status": "RELAY_OOF_RISK_PHYSICAL_DEVELOPMENT_NOT_SUCCESS",
    }
    single = {
        "risk_variant": "single",
        "status": "RELAY_SINGLE_RISK_PHYSICAL_DEVELOPMENT_NOT_SUCCESS",
    }
    assert study_predictions(oof, risk) == [0]
    assert study_predictions(single, risk) == [1]
    with pytest.raises(ValueError):
        study_predictions({**single, "status": oof["status"]}, risk)


def arrays():
    return {
        k: np.zeros((2,), dtype=np.float64)
        for k in (
            "qpos",
            "qvel",
            "torque",
            "ball_forces",
            "executed_receiver_action",
            "passer_action",
            "receiver_observation",
            "receiver_command",
            "receiver_return_action",
            "receiver_return_observation",
            "receiver_return_target",
            "passer_follow_action",
            "passer_follow_command",
            "passer_follow_control",
            "passer_follow_detector_delta",
            "passer_follow_observation",
            "passer_follow_target",
        )
    }


def test_exact_requires_all_actor_history_and_identical_physics():
    a, b = arrays(), arrays()
    assert exact_reference_arrays(a, b)
    b["qvel"][0] = 1
    assert not exact_reference_arrays(a, b)
    b = arrays()
    b["receiver_observation"][0] = np.nan
    assert not exact_reference_arrays(a, b)
    del b["passer_action"]
    with pytest.raises(ValueError):
        exact_reference_arrays(a, b)


def test_legacy_follow_fields_are_explicitly_unavailable_not_assumed():
    a, b = arrays(), arrays()
    for key in list(b):
        if key.startswith("passer_follow_"):
            del b[key]
            a[key] = np.zeros((0,), dtype=np.float64)
    assert exact_reference_arrays(a, b, legacy_empty_follow_unavailable=True)
    a["passer_follow_action"] = np.ones(1)
    assert not exact_reference_arrays(a, b, legacy_empty_follow_unavailable=True)
