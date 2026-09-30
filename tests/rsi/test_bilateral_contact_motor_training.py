import copy

import pytest

from scripts.rsi_train_bilateral_contact_motor_v303 import score, write_once


def row(*, clean=True, high=True, excursion=1, pelvis=0.7, reward=2):
    return {
        "clean_foot_only": clean,
        "high_quality": high,
        "maximum_lateral_excursion_m": excursion,
        "minimum_pelvis_z_m": pelvis,
        "reward": reward,
    }


def test_safety_retention_and_cleanliness_cannot_be_bought_with_reward():
    baseline = [row(), row(), row(clean=False, high=False)]
    good = [row(), row(), row()]
    for field, value in (
        ("minimum_pelvis_z_m", 0.64),
        ("clean_foot_only", False),
        ("maximum_lateral_excursion_m", 4.1),
    ):
        bad = copy.deepcopy(good)
        bad[0][field] = value
        bad[0]["reward"] = 1000
        assert score(good, baseline) > score(bad, baseline)
    bad = copy.deepcopy(good)
    bad[2]["high_quality"] = False
    bad[2]["reward"] = 1000
    assert score(good, baseline) > score(bad, baseline)


def test_resume_normalizes_json_sequences_and_rejects_different_commitment(tmp_path):
    path = tmp_path / "checkpoint.json"
    write_once(path, {"paired": [(1, 2)]})
    write_once(path, {"paired": [(1, 2)]})
    with pytest.raises(ValueError):
        write_once(path, {"paired": [(1, 3)]})
