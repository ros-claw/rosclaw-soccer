"""SIM_ONLY parent-contact option selection fails closed on invalid evidence."""

import pytest

from rosclaw_soccer.rsi.parent_contact_approach_gate import choose_shadow_approach_gain
from rosclaw_soccer.sim.contracts import hash_json


def _report(indices: list[int], pelvis: float = 0.7) -> dict:
    report = {
        "activation_ceiling": "SIM_ONLY",
        "navigation_lateral_ball_gain": 1.2,
        "navigation_lateral_negative_only": True,
        "frames": 300,
        "environments": [{"contact_body_indices": indices, "minimum_pelvis_z_m": pelvis}],
    }
    report["report_hash"] = hash_json(report)
    return report


def test_right_support_knee_veto_and_clean_candidate() -> None:
    assert choose_shadow_approach_gain(_report([0, 4, 5])) == (
        0.8,
        "right_support_knee_contact",
    )
    assert choose_shadow_approach_gain(_report([0, 4])) == (
        1.2,
        "candidate_parent_no_right_knee_contact",
    )


def test_absent_contact_and_low_pelvis_fallback() -> None:
    assert choose_shadow_approach_gain(_report([]))[0] == 0.8
    assert choose_shadow_approach_gain(_report([0], pelvis=0.64))[0] == 0.8


def test_forged_or_real_parent_rejected() -> None:
    forged = _report([0])
    forged["environments"][0]["contact_body_indices"].append(5)
    with pytest.raises(ValueError, match="sealed"):
        choose_shadow_approach_gain(forged)
    real = _report([0])
    real["activation_ceiling"] = "REAL"
    real["report_hash"] = hash_json({k: v for k, v in real.items() if k != "report_hash"})
    with pytest.raises(ValueError, match="sealed"):
        choose_shadow_approach_gain(real)
