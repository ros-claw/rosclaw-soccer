import copy

import pytest

from rosclaw_soccer.rsi.receiver_probe_trust_update import development_trust_update


def report(candidate_speed):
    measurement = lambda speed: {  # noqa: E731
        "first_contact": {"first_contact_groups": ["left_foot"]},
        "final_relative_speed_mps": speed,
        "final_root_ball_distance_m": 1.0,
    }
    return {
        "rows": [
            dict(
                index=i,
                context=i,
                side=0.0,
                baseline=measurement(2.0),
                candidate=measurement(candidate_speed),
                common_controls=36,
                same_exposure=True,
                baseline_body_failure=False,
                candidate_body_failure=False,
                clean_reception_certified=False,
            )
            for i in range(8)
        ]
    }


def test_quadratic_proposal_bounded_and_not_policy_promotion():
    result = development_trust_update(report(1.0), report(3.2))
    assert 0 < result["knee_proposal_rad"] <= 0.1
    assert result["neural_weight_updates"] == 0
    assert result["promotion_authorized"] is False


def test_unequal_exposure_or_baseline_rejected():
    a, b = report(1.0), report(3.0)
    c = copy.deepcopy(b)
    c["rows"][0]["same_exposure"] = False
    with pytest.raises(ValueError, match="identical complete"):
        development_trust_update(a, c)
    b["rows"][0]["baseline"]["final_relative_speed_mps"] = 9.0
    with pytest.raises(ValueError, match="identical complete"):
        development_trust_update(a, b)
