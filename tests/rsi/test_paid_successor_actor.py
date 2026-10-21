import copy

import pytest
import torch
from test_paid_live_distilled_actor import evidence

from rosclaw_soccer.rsi.paid_successor_actor import sequence_config, validate_successor_state


def sequence_evidence(variant=0):
    state, receipt, initial = evidence()
    state.update(
        schema="soccer.paid_successor_sequences.v1",
        config=sequence_config(variant),
        counts=dict(first_forward_rows=975360, second_forward_rows=975360, optimizer_updates=1024),
        bank_hash="bank",
    )
    for value in state["optimizer"]["state"].values():
        value["step"] = torch.tensor(1024.0)
    receipt.update(
        status="PAID_SUCCESSOR_SEQUENCE_TRAINED_NOT_PHYSICALLY_EXAMINED",
        config=copy.deepcopy(state["config"]),
        counts=copy.deepcopy(state["counts"]),
        CPU_whole_world_exam_required=True,
    )
    receipt["input_file_hashes"]["bank"] = "bank"
    return state, receipt, initial


@pytest.mark.parametrize("variant", range(4))
def test_separate_full_world_protocol_admitted_without_authority(variant):
    s, r, i = sequence_evidence(variant)
    validate_successor_state(s, r, i, declaration_hash="decl", scene_hash="scene")
    assert s["config"]["memory_handoff"] == ("reset" if variant == 3 else "carry")


@pytest.mark.parametrize(
    "fault",
    [
        "schema",
        "smoke",
        "counter",
        "memory",
        "DEV",
        "normalizer",
        "shape",
        "nan",
        "slot",
        "step",
        "bank",
        "authority",
    ],
)
def test_untrained_incompatible_or_corrupt_sequence_actor_rejected(fault):
    s, r, i = sequence_evidence()
    if fault == "schema":
        s["schema"] = "soccer.paid_live_distilled.v2"
    elif fault == "smoke":
        s["epoch"] = 1
    elif fault == "counter":
        s["counts"]["optimizer_updates"] = 4
    elif fault == "memory":
        s["config"]["memory_handoff"] = "reset"
        r["config"] = copy.deepcopy(s["config"])
    elif fault == "DEV":
        s["config"]["DEV_episodes_used"] = 16
    elif fault == "normalizer":
        s["network"]["mean"][0, 0] = 1
    elif fault == "shape":
        s["network"]["p0"] = torch.zeros(2)
    elif fault == "nan":
        s["network"]["p0"][0] = float("nan")
    elif fault == "slot":
        s["optimizer"]["state"][0]["exp_avg_sq"][0] = -1
    elif fault == "step":
        s["optimizer"]["state"][0]["step"] = torch.tensor(768)
    elif fault == "bank":
        s["bank_hash"] = "unbound"
    else:
        r["hardware_authorized"] = True
    with pytest.raises(ValueError):
        validate_successor_state(s, r, i, declaration_hash="decl", scene_hash="scene")


@pytest.mark.parametrize("variant", range(4))
def test_specialist_protocol_preserves_negative_body_anchors_not_success_labels(variant):
    s, r, i = sequence_evidence(variant)
    s.update(
        schema="soccer.paid_successor_specialist.v1",
        config=sequence_config(variant, specialist=True),
    )
    r.update(
        status="PAID_SUCCESSOR_SPECIALIST_TRAINED_NOT_PHYSICALLY_EXAMINED",
        config=copy.deepcopy(s["config"]),
    )
    validate_successor_state(s, r, i, declaration_hash="decl", scene_hash="scene")
    assert len(s["config"]["positive_successor_cases"]) == 14
    assert s["config"]["negative_body_only_cases"] == [10, 14]
    assert s["config"]["memory_handoff"] == "reset"
    s["config"]["positive_successor_cases"].append(10)
    r["config"] = copy.deepcopy(s["config"])
    with pytest.raises(ValueError):
        validate_successor_state(s, r, i, declaration_hash="decl", scene_hash="scene")
