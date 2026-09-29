"""Sealed SIM_ONLY tree inference has no sklearn or motor runtime dependency."""

import json

import numpy as np
import pytest

from rosclaw_soccer.rsi.team_proprio_tree_chooser import TeamProprioTreeChooser
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json


def _model(tmp_path):
    forest = tmp_path / "forest.npz"
    probability = np.full((720, 6), 0.1, dtype=np.float64)
    probability[:240] = 0.9
    probability[240:480] = 0.4
    probability[480:, 0] = 0.8
    np.savez_compressed(
        forest,
        tree_offsets=np.arange(721, dtype=np.int64),
        forest_offsets=np.arange(7, dtype=np.int64) * 120,
        feature=np.full(720, -2, dtype=np.int16),
        threshold=np.full(720, -2.0),
        left=np.full(720, -1, dtype=np.int64),
        right=np.full(720, -1, dtype=np.int64),
        positive_probability=probability,
    )
    arms = [f"arm_{index}" for index in range(6)]
    manifest = {
        "schema": "rsi_team_proprio_tree_chooser_model_v47",
        "activation_ceiling": "SIM_ONLY",
        "promotion_authorized": False,
        "feature_count": 89,
        "feature_set": "full_proprio_89",
        "forest_file": "forest.npz",
        "forest_hash": hash_bytes(forest.read_bytes()),
        "safety_threshold": 0.75,
        "safety_quantile": 0.2,
        "estimators_per_forest": 120,
        "ensemble_seeds": [11, 19],
        "arm_names": arms,
        "arm_parameters": {arm: {} for arm in arms},
    }
    manifest["model_hash"] = hash_json(manifest)
    path = tmp_path / "model.json"
    path.write_text(json.dumps(manifest))
    return path, forest


def test_numpy_tree_model_selects_safest_useful_arm(tmp_path):
    path, _ = _model(tmp_path)
    chooser = TeamProprioTreeChooser(
        agent_id="red.playmaker",
        foundation_hash="sha256:" + "a" * 64,
        foundation_config_hash="sha256:" + "b" * 64,
        model_path=path,
    )
    scores = chooser.predict_scores(np.zeros(89))
    assert scores.shape == (6, 3)
    assert scores[0, 0] == pytest.approx(0.9)
    assert scores[0, 2] == pytest.approx(0.8)
    assert chooser.choose_feature(np.zeros(89)) == "arm_0"
    with pytest.raises(ValueError):
        chooser.choose_feature(np.zeros(88))


def test_numpy_tree_model_rejects_tampering_and_nonfinite_weights(tmp_path):
    path, forest = _model(tmp_path)
    with np.load(forest, allow_pickle=False) as source:
        arrays = {key: np.asarray(source[key]) for key in source.files}
    arrays["positive_probability"][0, 0] = float("nan")
    np.savez_compressed(forest, **arrays)
    with pytest.raises(ValueError):
        TeamProprioTreeChooser(
            agent_id="red.playmaker",
            foundation_hash="sha256:" + "a" * 64,
            foundation_config_hash="sha256:" + "b" * 64,
            model_path=path,
        )
