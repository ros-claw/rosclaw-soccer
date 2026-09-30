import copy
import json

import numpy as np
import pytest

from rosclaw_soccer.rsi.contact_motor_primitive import make_policy
from rosclaw_soccer.sim.contracts import hash_json
from scripts.rsi_train_bilateral_contact_motor_v303 import load_warm_start, score, write_once


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


def test_warm_start_binds_complete_predecessor_without_promoting_it(tmp_path):
    commitment = {"activation_ceiling": "SIM_ONLY", "experiment": 303}
    policy = make_policy(np.full((3, 12), 0.02), hash_json(commitment))
    policy_path = tmp_path / "policy.json"
    write_once(policy_path, policy)
    summary = {
        "schema": "rsi_bilateral_contact_motor_training_v303",
        "activation_ceiling": "SIM_ONLY",
        "best": {"policy": str(policy_path)},
        "commitment": commitment,
        "independent_physical_episode_count": 84,
        "promotion_authorized": False,
        "consumed_development_gate_passed": False,
    }
    summary["report_hash"] = hash_json(summary)
    summary_path = tmp_path / "summary.json"
    write_once(summary_path, summary)
    loaded, knots, previous_hash = load_warm_start(policy_path, summary_path)
    assert loaded["promotion_authorized"] is False
    assert np.all(knots == 0.02)
    assert previous_hash == summary["report_hash"]
    for key, value in (
        ("independent_physical_episode_count", 83),
        ("promotion_authorized", True),
        ("activation_ceiling", "REAL"),
    ):
        bad = copy.deepcopy(summary)
        bad[key] = value
        bad["report_hash"] = hash_json({k: v for k, v in bad.items() if k != "report_hash"})
        summary_path.write_text(json.dumps(bad))
        with pytest.raises(ValueError):
            load_warm_start(policy_path, summary_path)
