import copy

from scripts.rsi_collect_bilateral_motor_validation_v305 import COURSES, score


def rows():
    result = []
    for i, (seed, lane) in enumerate(COURSES):
        arms = {}
        for arm in ("gain_08", "gain_12", "learned"):
            high = (
                i in (1, 6)
                if arm == "gain_08"
                else i >= 7
                if arm == "gain_12"
                else i in (1, 6) or i >= 7
            )
            arms[arm] = {
                "high_quality": high,
                "clean_foot_only": high,
                "maximum_lateral_excursion_m": 3,
                "minimum_pelvis_z_m": 0.7,
            }
        result.append({"seed": seed, "lane": lane, "arms": arms})
    return result


def test_all_consumed_cases_required_and_order_does_not_change_retention():
    good = rows()
    assert score(good)["consumed_validation_gate_passed"]
    assert score(list(reversed(good))) == score(good)
    assert not score(good[:-1])["consumed_validation_gate_passed"]
    assert not score(good + [good[0]])["consumed_validation_gate_passed"]


def test_old_success_gain_case_safety_or_out_cannot_be_ignored():
    for index, key, value in (
        (1, "high_quality", False),
        (7, "high_quality", False),
        (2, "minimum_pelvis_z_m", 0.64),
        (2, "maximum_lateral_excursion_m", 4.1),
        (6, "clean_foot_only", False),
    ):
        bad = copy.deepcopy(rows())
        bad[index]["arms"]["learned"][key] = value
        assert not score(bad)["consumed_validation_gate_passed"]
