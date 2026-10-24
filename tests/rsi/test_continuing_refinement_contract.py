import copy

import pytest
from test_parallel_refinement_batch import fixture

from rosclaw_soccer.rsi.continuing_refinement_contract import (
    FULL_KIND,
    KIND,
    continuing_config,
    contract_for,
    full_recovery_config,
    select_parallel_experience,
    validate_batch_coverage,
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


def test_full_recovery_is_new_contract_not_legacy_relabelling():
    from rosclaw_soccer.rsi.refinement_actor_clock import FULL

    cfg = full_recovery_config(25)
    c = contract_for(cfg)
    assert c.kind == FULL_KIND and c.prior_steps == 12 and c.worlds == 96
    assert c.schema.endswith(".v5")
    batch = [
        dict(
            controller_kind=FULL,
            observations=range(389),
            motion_reference_controls=289,
            observed_recovery_controls=100,
            total_observed_controls=389,
            deterministic_recovery_excluded_from_actor_rows=False,
            termination="FULL_MOTION_PLUS100NN_RECOVERY",
        )
        for _ in range(96)
    ]
    validate_batch_coverage(cfg, batch)
    with pytest.raises(ValueError):
        validate_batch_coverage(continuing_config(25, 3, 12), batch)
    batch[0] = batch[0] | dict(controller_kind="pure-refinement-stochastic-native-v1")
    with pytest.raises(ValueError):
        validate_batch_coverage(cfg, batch)
    with pytest.raises(ValueError):
        contract_for(cfg | dict(return_rule="legacy_delayed_recovery"))


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


def test_full96_selection_requires_new_version_all_native_and_law_not_old_tail():
    from rosclaw_soccer.rsi.parallel_refinement_batch import (
        FULL_OWN,
        FULL_SOURCE,
        pair_plan,
        parallel_config,
    )
    from rosclaw_soccer.rsi.refinement_actor_clock import FULL

    s, _ = reports()
    courses = [s["records"][i]["course"] for i in (0, 32, 64)]
    s.update(config=parallel_config(25, 32, full_recovery=True), status=FULL_SOURCE)
    for pair in range(48):
        for plan in pair_plan(s["config"], courses, pair):
            s["records"][plan["index"]].update(
                plan,
                controller_kind=FULL,
                stochastic_controls=389,
                deterministic_recovery_excluded_from_actor_rows=False,
            )
    o = copy.deepcopy(s)
    o.update(
        status=FULL_OWN,
        source_hash="source",
        sampling_law_audit_passed=True,
        independent_whole_world_replay=True,
    )
    assert (
        len(
            select_parallel_experience(
                s, o, source_hash="source", training_hash="parent", state=25, full_recovery=True
            )
        )
        == 96
    )
    with pytest.raises(ValueError):
        select_parallel_experience(s, o, source_hash="source", training_hash="parent", state=25)
    o["sampling_law_audit_passed"] = False
    with pytest.raises(ValueError):
        select_parallel_experience(
            s, o, source_hash="source", training_hash="parent", state=25, full_recovery=True
        )


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
