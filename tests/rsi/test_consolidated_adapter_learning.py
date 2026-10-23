"""Metadata tests use synthetic Adam counts, not evidence of scientific training."""

import copy

import pytest
import torch
from test_consolidated_motor_adapter import learned_memory
from test_single_mode_pass_curriculum import records

from rosclaw_soccer.rsi.consolidated_adapter_learning import (
    SCHEMA,
    adapter_config,
    adapter_fit_plan,
    adapter_schedule,
    validate_head_checkpoint,
)
from rosclaw_soccer.rsi.consolidated_motor_adapter import ConsolidatedMotorAdapter


def test_all_original_skill_anchors_and_coherent_passes_have_full_coverage():
    rows = records()
    before = copy.deepcopy(rows)
    plan = adapter_fit_plan(rows)
    assert plan["old_anchor_indices"] == list(range(17))
    schedules = []
    for seed in (105023, 105024, 105025):
        schedule = adapter_schedule(plan, seed)
        assert len(schedule) == 2048
        assert {i for r in schedule for i in r["teacher_indices"]} == set(plan["teacher_indices"])
        assert {i for r in schedule for i in r["old_anchor_indices"]} == set(range(17))
        assert {r["baseline_pass_index"] for r in schedule} == set(plan["baseline_pass_indices"])
        schedules.append(schedule)
    assert schedules[0] != schedules[1] != schedules[2]
    assert rows == before


def synthetic_checkpoint():
    cfg = adapter_config(105023, "cpu")
    model = ConsolidatedMotorAdapter(learned_memory(), head_bound_rad=cfg["head_bound_rad"])
    initial = copy.deepcopy(dict(model.head.state_dict()))
    opt = torch.optim.Adam(model.plastic_parameters(), lr=cfg["head_lr"])
    for _ in range(2):
        opt.zero_grad()
        x = torch.randn(4, 1, 288)
        added, _ = model.adapt_features(x, torch.zeros(4, 1, 29), torch.ones(29))
        (added - 0.01).square().mean().backward()
        opt.step()
    optimizer = opt.state_dict()
    for slot in optimizer["state"].values():
        slot["step"].fill_(2048)  # fixture only, not claimed optimizer execution
    commitments = {
        k: "sha256:" + k
        for k in (
            "feature_hash",
            "data_hash",
            "fit_plan_hash",
            "memory_training_hash",
            "memory_checkpoint_hash",
            "parent_lineage_hash",
        )
    }
    state = dict(
        schema=SCHEMA,
        config=cfg,
        head=dict(model.head.state_dict()),
        initial_head=initial,
        head_optimizer=optimizer,
        head_optimizer_steps=2048,
        **commitments,
    )
    return state, dict(model.head.state_dict()), cfg, commitments


def test_separate_head_checkpoint_admits_exact_new_parameter_adam_ownership():
    s, expected, cfg, ids = synthetic_checkpoint()
    validate_head_checkpoint(s, expected, config=cfg, commitments=ids)


@pytest.mark.parametrize(
    "fault", ["counter", "slot", "moment", "unchanged", "extra", "ancestry", "initial"]
)
def test_head_admission_rejects_changed_authority_ancestry_or_false_training(fault):
    s, expected, cfg, ids = synthetic_checkpoint()
    s = copy.deepcopy(s)
    if fault == "counter":
        s["head_optimizer_steps"] = 0
    elif fault == "slot":
        s["head_optimizer"]["state"][0]["step"].fill_(112)
    elif fault == "moment":
        s["head_optimizer"]["state"][2]["exp_avg"].fill_(float("nan"))
    elif fault == "unchanged":
        s["head"]["0.weight"] = s["initial_head"]["0.weight"].clone()
    elif fault == "extra":
        s["old_memory_rewritten"] = True
    elif fault == "ancestry":
        s["memory_training_hash"] = "sha256:other"
    elif fault == "initial":
        s["initial_head"]["2.bias"].add_(0.01)
    with pytest.raises(ValueError):
        validate_head_checkpoint(s, expected, config=cfg, commitments=ids)


@pytest.mark.parametrize("bound", [float("nan"), float("inf"), True, 0, -0.1, 0.21])
def test_no_undeclared_expansion_of_the_total_motor_authority(bound):
    with pytest.raises(ValueError):
        ConsolidatedMotorAdapter(learned_memory(), head_bound_rad=bound)


def test_larger_required_new_head_still_respects_original_total_authority():
    model = ConsolidatedMotorAdapter(learned_memory(), head_bound_rad=0.2)
    with torch.no_grad():
        model.head[-1].bias.fill_(10)
    added, total = model.adapt_features(
        torch.zeros(3, 1, 288), torch.full((3, 1, 29), 0.19), torch.ones(29)
    )
    assert added.max() < 0.011
    assert (total.abs() <= 0.2).all()
