import copy

import pytest
from test_refinement_sampling_exam import fixture

from rosclaw_soccer.rsi.refinement_continual_batch import continual_config, select_learned_arm


def reports():
    source, _ = fixture()
    source.update(
        status="PPO_NEW_SEED_PAIRED48_COLLECTED_PENDING_OWN_NATIVE_LAW",
        training_hash="trained-parent",
        optimizer_updates=0,
        actor_training_admitted=False,
        promotion_authorized=False,
        hardware_authorized=False,
        goal_complete=False,
        Fresh_verified=False,
        team_game_verified=False,
    )
    own = copy.deepcopy(source)
    own.update(
        status="PPO_NEW_SEED_PAIRED48_ALL_OWN_NATIVE_AND_LAW_MATCH",
        source_hash="source",
        sampling_law_audit_passed=True,
        independent_whole_world_replay=True,
    )
    return source, own


def select(source, own):
    return select_learned_arm(source, own, source_hash="source", training_hash="trained-parent")


def test_all_learned_arm_only_no_control_or_hard_case_selection():
    s, o = reports()
    chosen = select(s, o)
    assert [r["index"] for r in chosen] == list(range(1, 48, 2))
    assert all(r["arm"] == "learned" for r in chosen)
    assert len(s["records"]) == 48
    assert s["actor_training_admitted"] is False


@pytest.mark.parametrize(
    "fault", ["parent", "source", "law", "own", "promote", "credit", "arm", "missing"]
)
def test_unqualified_or_offpolicy_data_rejected(fault):
    s, o = reports()
    if fault == "parent":
        s["training_hash"] = "different"
    elif fault == "source":
        o["source_hash"] = "different"
    elif fault == "law":
        o["sampling_law_audit_passed"] = False
    elif fault == "own":
        o["records"][1]["outcome"]["whole_body_healthy"] = False
    elif fault == "promote":
        s["promotion_authorized"] = True
    elif fault == "credit":
        s["optimizer_updates"] = 1
    elif fault == "arm":
        s["records"][1]["arm"] = "old"
    else:
        s["records"].pop()
    with pytest.raises(ValueError):
        select(s, o)


def test_generation1_declares_trained_critic_and_moment_carry_without_success_claim():
    c = continual_config(25)
    assert c["generation_index"] == 1 and c["parent_generation_index"] == 0
    assert c["advantage_rule"] == "MONTE_CARLO_MINUS_FROZEN_PRIOR_LEARNED_CRITIC_GLOBAL_STD"
    assert (
        c["optimizer_initialization"] == "CARRY_PRIOR_ADAM_MOMENTS_COUNTERS_REBASE_LR_PER_NEW_BATCH"
    )
    assert c["actor_lr"] == 3e-5 and c["critic_lr"] == 0.001
    assert c["continuous_learning_proven"] is False
