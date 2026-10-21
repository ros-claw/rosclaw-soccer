import copy

import pytest
import torch

from rosclaw_soccer.rsi.paid_contact_residual import ContactResidual, residual_config
from rosclaw_soccer.rsi.paid_contact_residual_actor import validate_contact_state
from rosclaw_soccer.rsi.paid_trainable_actor import PUBLISHED_HASH


def evidence(variant=0):
    template = ContactResidual().state_dict()
    config = residual_config(variant)
    counts = dict(rescue_forward_rows=377344, zero_forward_rows=598016, optimizer_updates=1024)
    flags = dict(
        hardware_authorized=False, promotion_authorized=False, commercial_publicity_authorized=False
    )
    state = dict(
        schema="soccer.paid_contact_residual.v1",
        epoch=256,
        config=config,
        counts=counts,
        declaration_hash="decl",
        body_scene_hash="scene",
        original_actor_hash=PUBLISHED_HASH,
        source_head="head",
        residual=copy.deepcopy(template),
        optimizer=dict(
            param_groups=[dict(params=list(range(10)), lr=0.0003)],
            state={
                i: dict(
                    step=torch.tensor(1024.0),
                    exp_avg=torch.zeros_like(value),
                    exp_avg_sq=torch.zeros_like(value),
                )
                for i, value in enumerate(template.values())
            },
        ),
        **flags,
    )
    receipt = dict(
        status="PAID_CONTACT_RESIDUAL_TRAINED_NOT_PHYSICALLY_EXAMINED",
        source_head="head",
        config=copy.deepcopy(config),
        counts=copy.deepcopy(counts),
        native_steps=0,
        CPU_zero_neural_replay_rows=7620,
        CPU_feature_recompute_rows=3810,
        **flags,
    )
    return state, receipt, template


@pytest.mark.parametrize("variant", range(4))
def test_separate_gated_residual_protocol_admitted_without_base_optimizer(variant):
    s, r, t = evidence(variant)
    validate_contact_state(s, r, t, declaration_hash="decl", scene_hash="scene")
    assert len(s["optimizer"]["state"]) == 10


@pytest.mark.parametrize(
    "fault",
    [
        "schema",
        "smoke",
        "count",
        "base_optimizer",
        "weight",
        "nan",
        "dtype",
        "shape",
        "Adam",
        "step",
        "body",
        "authority",
        "zero_proof",
        "features",
        "labels",
    ],
)
def test_incompatible_untrained_or_unsafe_residual_rejected(fault):
    s, r, t = evidence()
    if fault == "schema":
        s["schema"] = "soccer.paid_successor_specialist.v1"
    elif fault == "smoke":
        s["epoch"] = 1
    elif fault == "count":
        s["counts"]["optimizer_updates"] = 4
    elif fault == "base_optimizer":
        s["optimizer"]["state"][10] = {}
    elif fault == "weight":
        s["config"]["retention_weight"] = 1
    elif fault == "nan":
        s["residual"]["net.4.bias"][0] = float("nan")
    elif fault == "dtype":
        s["residual"]["net.4.bias"] = s["residual"]["net.4.bias"].double()
    elif fault == "shape":
        s["residual"]["net.4.bias"] = torch.zeros(1)
    elif fault == "Adam":
        s["optimizer"]["state"][0]["exp_avg_sq"].fill_(-1)
    elif fault == "step":
        s["optimizer"]["state"][0]["step"] = torch.tensor(768.0)
    elif fault == "body":
        s["body_scene_hash"] = "other"
    elif fault == "authority":
        r["promotion_authorized"] = True
    elif fault == "zero_proof":
        r["CPU_zero_neural_replay_rows"] = 0
    elif fault == "features":
        s["config"]["residual_features"] = 160
    else:
        s["config"]["old_good_zero_cases"].append(10)
    with pytest.raises(ValueError):
        validate_contact_state(s, r, t, declaration_hash="decl", scene_hash="scene")


def projected_evidence(variant=0):
    s, r, t = evidence(variant)
    s["schema"] = "soccer.paid_contact_residual.v2"
    s["config"] = residual_config(variant, projected=True)
    calls = 1024 if variant < 3 else 0
    cert = dict(
        negative_support_rows=3000,
        maximum_logit_cpu=-0.2 if variant < 3 else 0.4,
        feasible=variant < 3,
        projection=dict(
            calls=calls,
            gate_evaluations=calls,
            rejections=0,
            accepted_fractions={"1.0": calls} if calls else {},
            negative_support_rows=3000,
        ),
    )
    s["counts"].update(
        hard_negative_gate_rows=3000 * 1024,
        projection_gate_rows=3000 * calls,
        CPU_constraint_gate_rows=15000,
    )
    s["retention_certificate"] = cert
    r.update(
        config=copy.deepcopy(s["config"]),
        counts=copy.deepcopy(s["counts"]),
        retention_certificate=copy.deepcopy(cert),
    )
    return s, r, t


@pytest.mark.parametrize("variant", range(4))
def test_projected_protocol_counts_actual_gate_work_and_keeps_explicit_ablation(variant):
    s, r, t = projected_evidence(variant)
    validate_contact_state(s, r, t, declaration_hash="decl", scene_hash="scene")
    assert s["config"]["gate_projection"] == (variant < 3)


@pytest.mark.parametrize(
    "fault",
    [
        "schema",
        "count",
        "rows",
        "projection_count",
        "fractions",
        "max_score",
        "false_feasible",
        "receipt",
        "warm_config",
        "bool_counter",
    ],
)
def test_projected_work_or_feasibility_cannot_masquerade_as_v1_or_pass_by_summary(fault):
    s, r, t = projected_evidence()
    if fault == "schema":
        s["schema"] = "soccer.paid_contact_residual.v1"
    elif fault == "count":
        s["counts"]["hard_negative_gate_rows"] += 1
    elif fault == "rows":
        s["retention_certificate"]["negative_support_rows"] = 1
    elif fault == "projection_count":
        s["retention_certificate"]["projection"]["calls"] = 0
    elif fault == "fractions":
        s["retention_certificate"]["projection"]["accepted_fractions"] = {"1.5": 1024}
    elif fault == "max_score":
        s["retention_certificate"]["maximum_logit_cpu"] = float("nan")
    elif fault == "false_feasible":
        s["retention_certificate"]["maximum_logit_cpu"] = 0.2
    elif fault == "receipt":
        r["retention_certificate"]["projection"]["rejections"] = 1
    elif fault == "warm_config":
        s["config"]["warmstart"] = "random_or_uncertified"
    else:
        s["retention_certificate"]["projection"]["calls"] = True
    with pytest.raises(ValueError):
        validate_contact_state(s, r, t, declaration_hash="decl", scene_hash="scene")
