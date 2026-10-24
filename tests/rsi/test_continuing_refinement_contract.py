import copy

import pytest
from test_parallel_refinement_batch import fixture

from rosclaw_soccer.rsi.continuing_refinement_contract import (
    KIND,
    continuing_config,
    contract_for,
    select_parallel_experience,
)
from rosclaw_soccer.rsi.refinement_continual_batch import continual_config


def reports():
    source, _ = fixture()
    source.update(
        status="GEN1_PARALLEL_OWN_POLICY_BATCH_COLLECTED_PENDING_NATIVE_LAW",
        training_hash="parent",
        optimizer_updates=0,
        actor_training_admitted=False,
        hardware_authorized=False,
        promotion_authorized=False,
        goal_complete=False,
        Fresh_verified=False,
        team_game_verified=False,
    )
    own = copy.deepcopy(source)
    own.update(
        status="GEN1_PARALLEL_OWN_POLICY_BATCH_ALL_NATIVE_LAW_MATCH",
        source_hash="source",
        sampling_law_audit_passed=True,
        independent_whole_world_replay=True,
    )
    return source, own


def test_legacy_and_continuing_contracts_stay_distinct():
    old = contract_for(continual_config(25))
    new = contract_for(continuing_config(25, 2, 8))
    assert (
        old.worlds == 24 and old.prior_steps == 4 and old.selected_indices == tuple(range(1, 48, 2))
    )
    assert new.worlds == 96 and new.prior_steps == 8 and new.selected_indices == tuple(range(96))
    assert new.kind == KIND and old.kind != KIND


@pytest.mark.parametrize(
    "generation,steps", [(1, 4), (2, 4), (2, 9), (3, 5), (17, 68), (True, 8), (2, 8.0)]
)
def test_invalid_or_reset_continuation_contract_rejected(generation, steps):
    with pytest.raises(ValueError):
        continuing_config(25, generation, steps)


def test_all96_new_same_policy_rows_not_a_24row_or_control_arm_subset():
    s, o = reports()
    selected = select_parallel_experience(
        s, o, source_hash="source", training_hash="parent", state=25
    )
    assert [r["index"] for r in selected] == list(range(96))
    assert len(s["records"]) == 96 and s["actor_training_admitted"] is False


@pytest.mark.parametrize(
    "fault", ["partial", "parent", "source", "law", "own", "arm", "authority", "credit"]
)
def test_unclosed_mixed_or_offpolicy_batches_cannot_train(fault):
    s, o = reports()
    if fault == "partial":
        s["records"].pop()
    elif fault == "parent":
        s["training_hash"] = "other"
    elif fault == "source":
        o["source_hash"] = "other"
    elif fault == "law":
        o["sampling_law_audit_passed"] = False
    elif fault == "own":
        o["records"][0]["tampered"] = True
    elif fault == "arm":
        s["records"][0]["arm"] = "old"
    elif fault == "authority":
        s["hardware_authorized"] = True
    else:
        o["optimizer_updates"] = 1
    with pytest.raises(ValueError):
        select_parallel_experience(s, o, source_hash="source", training_hash="parent", state=25)
