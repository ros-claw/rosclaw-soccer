import copy

import pytest

from scripts.rsi_train_failure_curriculum_motor_v306 import rank


def reference():
    rows = []
    for index in range(12):
        arms = {}
        for arm in ("gain_08", "gain_12", "learned"):
            high = (
                index in (1, 6)
                if arm == "gain_08"
                else index >= 7
                if arm == "gain_12"
                else index in (4, 6) or index >= 7
            )
            arms[arm] = {
                "high_quality": high,
                "clean_foot_only": high or index in (0, 3, 5),
                "maximum_lateral_excursion_m": 3,
                "minimum_pelvis_z_m": 0.7,
                "reward": 2,
            }
        rows.append({"arms": arms})
    return rows


def test_protected_successes_and_safety_beat_large_reward_or_new_success_count():
    ref = reference()
    good = [copy.deepcopy(r["arms"]["learned"]) for r in ref]
    for index, key, value in (
        (0, "minimum_pelvis_z_m", 0.64),
        (7, "high_quality", False),
        (0, "clean_foot_only", False),
        (2, "maximum_lateral_excursion_m", 4.1),
    ):
        bad = copy.deepcopy(good)
        bad[index][key] = value
        bad[index]["reward"] = 10000
        assert rank(good, ref) > rank(bad, ref)


def test_recovering_an_old_anchor_is_rewarded_without_forgetting_other_skills():
    ref = reference()
    base = [copy.deepcopy(r["arms"]["learned"]) for r in ref]
    candidate = copy.deepcopy(base)
    candidate[1]["high_quality"] = True
    candidate[1]["clean_foot_only"] = True
    assert rank(candidate, ref) > rank(base, ref)
    with pytest.raises(ValueError):
        rank(candidate[:-1], ref)
