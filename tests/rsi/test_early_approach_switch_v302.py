"""A missing or reordered course must not become a passing mechanism trial."""

from scripts.rsi_collect_early_approach_switch_v302 import COURSES, score


def row(index: int) -> dict:
    seed, lane = COURSES[index]
    return {
        "seed": seed,
        "lane": lane,
        "response": {"max_root_xy_delta_before_contact_m": 0.03},
        "arms": {
            "gain_08": {"clean_foot_only": True, "maximum_lateral_excursion_m": 3.0},
            "gain_12": {"clean_foot_only": False, "maximum_lateral_excursion_m": 4.5},
            "early_switch_10": {
                "clean_foot_only": True,
                "maximum_lateral_excursion_m": 3.0,
                "high_quality": index == 2,
            },
        },
    }


def test_mechanism_score_binds_roles_by_course() -> None:
    assert score([row(2), row(0), row(1)], []) == score([row(0), row(1), row(2)], [])
    assert score([row(2), row(0), row(1)], [])["mechanism_gate_passed"] is True


def test_missing_risk_course_is_not_replaced_by_gain() -> None:
    result = score([row(2), row(0)], [])
    assert result["risk_rescued_count"] == 1
    assert result["complete"] is False
    assert result["mechanism_gate_passed"] is False
