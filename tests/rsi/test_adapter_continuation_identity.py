import pytest

from rosclaw_soccer.rsi.adapter_continuation_identity import (
    continuation_course,
    continuation_protocol,
)


def legacy():
    return dict(
        kind="team-pass-consolidated-v1",
        seed=75024,
        training=dict(config=dict(seed=75024), data_hash="bank177", warm_start_hash="memory65023"),
        bank=dict(warm_start_hash="memory65023"),
        bank_hash="bank177",
    )


def adapter():
    a = legacy()
    a.update(
        kind="consolidated-adapter-v1",
        seed=105025,
        training=dict(
            config=dict(seed=105025), data_hash="bank274", memory_training_hash="memory65023"
        ),
        child_bank=dict(
            status="TEAM_PASS_ACTUAL_STUDENT_CORRECTION_BANK274_READY_NOT_TRAINED",
            base_data_hash="bank177",
            records=[{}] * 274,
            Fresh_fit_rows=0,
        ),
        child_bank_hash="bank274",
    )
    return a


def test_legacy_continuation_identity_and_protocol_do_not_change():
    assert (
        continuation_protocol(**legacy()) == "soccer.student_actual_prefix_expert_suffix_TRAIN80.v1"
    )


@pytest.mark.parametrize("seed", [105024, 105025])
def test_new_heads_have_explicit_new_ancestry_and_protocol(seed):
    a = adapter()
    a["seed"] = a["training"]["config"]["seed"] = seed
    assert continuation_protocol(**a) == "soccer.adapter_actual_prefix_expert_suffix_TRAIN80.v1"


@pytest.mark.parametrize(
    "fault",
    [
        "rejected_seed",
        "kind",
        "model_seed",
        "data",
        "memory",
        "base",
        "count",
        "fresh",
        "missing_child",
    ],
)
def test_new_kind_cannot_bypass_ancestry_or_old_loss_exclusion(fault):
    a = adapter()
    if fault == "rejected_seed":
        a["seed"] = a["training"]["config"]["seed"] = 105023
    elif fault == "kind":
        a["kind"] = "duck-typed-wrapper"
    elif fault == "model_seed":
        a["training"]["config"]["seed"] = 105024
    elif fault == "data":
        a["training"]["data_hash"] = "different"
    elif fault == "memory":
        a["training"]["memory_training_hash"] = "different"
    elif fault == "base":
        a["child_bank"]["base_data_hash"] = "different"
    elif fault == "count":
        a["child_bank"]["records"].pop()
    elif fault == "fresh":
        a["child_bank"]["Fresh_fit_rows"] = 1
    else:
        a["child_bank"] = None
    with pytest.raises(ValueError):
        continuation_protocol(**a)


def test_new_model_never_enters_through_legacy_kind():
    a = adapter()
    a["kind"] = "team-pass-consolidated-v1"
    with pytest.raises(ValueError):
        continuation_protocol(**a)


def test_self_based_stroke_is_a_separate_new_protocol_not_a_legacy_switch():
    a = adapter()
    a["teacher_basis"] = "student"
    assert continuation_protocol(**a) == "soccer.adapter_self_based_stroke_TRAIN80.v1"
    a = legacy()
    a["teacher_basis"] = "student"
    with pytest.raises(ValueError):
        continuation_protocol(**a)


def test_self_based_calibration_changes_only_control_arm_latent_not_initial_state():
    c = dict(
        latent=[[0.2] * 6 for _ in range(2)],
        side=0.55,
        context=20,
        assisted=True,
        target=[0, 1, 0.11],
    )
    assert continuation_course(c, 0, "student")["latent"] == [[0.0] * 6 for _ in range(2)]
    assert continuation_course(c, 80, "student") == c
    assert continuation_course(c, 0, "retained65023") == c
    assert continuation_course(c, 80, "retained65023") == c
    assert c["latent"][0][0] == 0.2
    assert {k: v for k, v in continuation_course(c, 0, "student").items() if k != "latent"} == {
        k: v for k, v in c.items() if k != "latent"
    }
    with pytest.raises(ValueError):
        continuation_course(c, 81, "student")
    with pytest.raises(ValueError):
        continuation_course(c, 0, "guessed")
