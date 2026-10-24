import copy

import pytest
from test_joint_delivery_teacher import row
from test_refinement_paired_exam import fixture

from rosclaw_soccer.rsi.refinement_paired_review import paired_review


def reports(old_error=0.6, new_error=0.3):
    source, _ = fixture()
    source.update(
        status="PPO_PAIRED_MEAN12_COLLECTED_PENDING_OWN_NATIVE", training_hash="sha256:train"
    )
    for r in source["records"]:
        r["outcome"] = row(old_error if r["arm"] == "old" else new_error, 2.7)["outcome"]
    own = copy.deepcopy(source)
    own.update(status="PPO_PAIRED_MEAN12_ALL_OWN_NATIVE_MATCH", independent_whole_world_replay=True)
    return source, own


def test_local_reward_gain_is_not_strict_or_team_success():
    result = paired_review(*reports())
    assert result["local_research_screen_passed"] is True
    assert result["strict_gains"] == 0
    assert result["unique_consumed_TRAIN_initials"] == 3
    for k in (
        "goal_complete",
        "promotion_authorized",
        "Fresh_verified",
        "team_game_verified",
        "continuous_learning_proven",
    ):
        assert result[k] is False


def test_local_regression_goes_back_to_analysis_not_promotion():
    result = paired_review(*reports(old_error=0.3, new_error=0.6))
    assert result["local_research_screen_passed"] is False
    assert result["next_agent_action"] == "ANALYZE_ACTUAL_REGRESSIONS_BEFORE_NEXT_ONPOLICY_BATCH"


def test_one_safety_regression_is_not_hidden_by_average_error_gain():
    source, _ = reports()
    for i in (1, 3):
        source["records"][i]["outcome"]["whole_body_healthy"] = False
    own = copy.deepcopy(source)
    own.update(status="PPO_PAIRED_MEAN12_ALL_OWN_NATIVE_MATCH", independent_whole_world_replay=True)
    result = paired_review(source, own)
    assert result["safety_losses"] == 1 and not result["local_research_screen_passed"]


def test_all_own_rows_must_match():
    source, own = reports()
    for i in (1, 3):
        own["records"][i]["outcome"]["whole_body_healthy"] = False
    with pytest.raises(ValueError, match="independent"):
        paired_review(source, own)
