"""Admission metadata fixtures, not evidence that any real training happened."""

import copy
import json

import pytest
import torch
from test_shared_native_neural_player import players
from test_single_mode_pass_curriculum import records as coherent_records
from test_team_pass_correction import bank

from rosclaw_soccer.rsi.dual_motor_checkpoint import digest
from rosclaw_soccer.rsi.dual_recurrent_actor_critic import MotorCritic
from rosclaw_soccer.rsi.single_mode_pass_curriculum import single_mode_pass_curriculum
from rosclaw_soccer.rsi.team_pass_correction import (
    CORRECTION_BANK_STATUS,
    CORRECTION_STATUS,
    correction_config,
    correction_schedule,
)
from rosclaw_soccer.rsi.team_pass_correction_checkpoint import load_team_pass_correction
from rosclaw_soccer.rsi.team_pass_single_mode import STATUS as SELECTED_STATUS
from rosclaw_soccer.rsi.team_pass_single_mode import selected_config, selected_schedule


def fixture(tmp_path, monkeypatch, fault=None, stage="all-positive"):
    dataset = bank() if stage == "all-positive" else coherent_records()
    _, ps = players()
    model = ps["a/"].actor
    model.plastic.head = torch.nn.Sequential(
        torch.nn.Linear(128, 128),
        torch.nn.ELU(),
        torch.nn.Linear(128, 64),
        torch.nn.ELU(),
        torch.nn.Linear(64, 32),
        torch.nn.ELU(),
        torch.nn.Linear(32, 29),
    )
    critic = MotorCritic()
    params = list(model.plastic.parameters()) + list(critic.parameters())
    opt = torch.optim.Adam([dict(params=params[:16], lr=1e-5), dict(params=params[16:], lr=0.001)])
    for p in params:
        p.grad = torch.ones_like(p)
    opt.step()
    warm = dict(
        plastic_network=copy.deepcopy(dict(model.plastic.state_dict())),
        critic=copy.deepcopy(dict(critic.state_dict())),
        optimizer=copy.deepcopy(opt.state_dict()),
    )
    for i, v in warm["optimizer"]["state"].items():
        v["step"].fill_(128 if i < 16 else 48)
    wp = tmp_path / "warm.pt"
    torch.save(warm, wp)
    wr = tmp_path / "warm"
    wr.mkdir()
    rp = wr / "complete.json"
    rp.write_text(
        json.dumps(
            dict(
                config=dict(seed=75024),
                checkpoint_path=str(wp),
                checkpoint_hash=digest(wp),
                cumulative_optimizer_steps=128,
                cumulative_critic_optimizer_steps=48,
            )
        )
    )
    wh = digest(rp)
    data = tmp_path / "bank"
    data.mkdir()
    bp = data / "complete.json"
    bp.write_text(
        json.dumps(
            dict(
                status=CORRECTION_BANK_STATUS,
                records=dataset,
                warm_start_hash=wh,
                parent_lineage_hash="sha256:parent",
            )
        )
    )
    bh = digest(bp)
    root = tmp_path / "candidate"
    root.mkdir()
    first = root / "initial.pt"
    torch.save(warm, first)
    cfg = (
        correction_config(85023, "cpu")
        if stage == "all-positive"
        else selected_config(95023, "cpu")
    )
    last = copy.deepcopy(warm)
    for name, _ in model.plastic.named_parameters():
        last["plastic_network"][name].add_(0.001)
    for i, v in last["optimizer"]["state"].items():
        if i < 16:
            v["step"].fill_(192)
    last.update(
        schema="soccer.dual_motor.team_pass_correction.v2"
        if stage == "all-positive"
        else "soccer.dual_motor.team_pass_correction.v3",
        config=cfg,
        optimizer_updates=dict(actor=192, critic=48),
        warm_start_hash=wh,
        parent_lineage_hash="sha256:parent",
    )
    history = [
        x | dict(loss=0.1, imitation=0.1, retention=0.01, new_retention=0.01, gradient_norm=0.1)
        for x in (
            correction_schedule(dataset, 85023)
            if stage == "all-positive"
            else selected_schedule(dataset, 95023)
        )
    ]
    if fault == "critic":
        next(iter(last["critic"].values())).add_(0.001)
    elif fault == "slot":
        last["optimizer"]["state"][0]["step"].fill_(191)
    elif fault == "critic_slot":
        last["optimizer"]["state"][16]["exp_avg"].add_(0.001)
    elif fault == "schedule":
        history[0]["teacher_indices"][0] = 0
    elif fault == "buffer":
        name = next(
            k for k in last["plastic_network"] if k not in dict(model.plastic.named_parameters())
        )
        last["plastic_network"][name].add_(0.001)
    path = root / "final.pt"
    torch.save(last, path)
    pins = {str(rp): wh, str(bp): bh}
    d = dict(
        source_head="fixture",
        config=cfg,
        input_file_hashes=pins,
        warm_start_hash=wh,
        data_hash=bh,
        parent_lineage_hash="sha256:parent",
        goal_complete=False,
        hardware_authorized=False,
        promotion_authorized=False,
    )
    if stage == "single-mode":
        plan = single_mode_pass_curriculum(dataset)
        d["fit_selection"] = plan
        last["fit_selection_hash"] = plan["selection_hash"]
        torch.save(last, path)
        if fault == "selection":
            plan["selected_teacher_indices"][0] = 0
    dp = root / "declaration.json"
    dp.write_text(json.dumps(d))
    r = d | dict(
        status=CORRECTION_STATUS if stage == "all-positive" else SELECTED_STATUS,
        declaration_hash=digest(dp),
        offline_supervised_steps=64,
        history=history,
        initial_optimizer_steps=128,
        cumulative_optimizer_steps=192,
        initial_critic_optimizer_steps=48,
        cumulative_critic_optimizer_steps=48,
        PPO_optimizer_steps_this_stage=0,
        cumulative_PPO_actor_steps=16,
        physical_steps_executed=0,
        cpu_replay=[{} for _ in range(274)],
        online_RL_claim=False,
        old_retention_proven=False,
        initial_checkpoint_path=str(first),
        initial_checkpoint_hash=digest(first),
        checkpoint_path=str(path),
        checkpoint_hash=digest(path),
        changed_parameter_tensors=list(dict(model.plastic.named_parameters())),
    )
    complete = root / "complete.json"
    complete.write_text(json.dumps(r))
    monkeypatch.setattr(
        "rosclaw_soccer.rsi.team_pass_correction_checkpoint.load_dual_candidate_for_sim_audit",
        lambda *args, **kwargs: (model, {}),
    )
    return model, root, digest(complete)


def test_complete_new_phase_requires_frozen_critic_and_complete_adam_carry(tmp_path, monkeypatch):
    model, root, h = fixture(tmp_path, monkeypatch)
    actual, report = load_team_pass_correction(
        model.parent,
        parent_lineage_hash="sha256:parent",
        root=root,
        expected_receipt_hash=h,
        depth=0,
    )
    assert not actual.training and not any(p.requires_grad for p in actual.parameters())
    assert report["cumulative_optimizer_steps"] == 192


@pytest.mark.parametrize("fault", ["critic", "slot", "critic_slot", "schedule", "buffer"])
def test_changed_critic_counter_schedule_or_frozen_buffer_is_rejected(tmp_path, monkeypatch, fault):
    model, root, h = fixture(tmp_path, monkeypatch, fault)
    with pytest.raises(ValueError):
        load_team_pass_correction(
            model.parent,
            parent_lineage_hash="sha256:parent",
            root=root,
            expected_receipt_hash=h,
            depth=0,
        )


def test_v3_coherent_curriculum_is_explicit_and_cannot_load_as_v2(tmp_path, monkeypatch):
    model, root, h = fixture(tmp_path, monkeypatch, stage="single-mode")
    with pytest.raises(ValueError):
        load_team_pass_correction(
            model.parent,
            parent_lineage_hash="sha256:parent",
            root=root,
            expected_receipt_hash=h,
            depth=0,
        )
    actual, report = load_team_pass_correction(
        model.parent,
        parent_lineage_hash="sha256:parent",
        root=root,
        expected_receipt_hash=h,
        depth=0,
        stage="single-mode",
    )
    assert not actual.training and report["status"] == SELECTED_STATUS


def test_v3_fit_mask_must_be_exact_recomputation_not_edited_outcomes(tmp_path, monkeypatch):
    model, root, h = fixture(tmp_path, monkeypatch, fault="selection", stage="single-mode")
    with pytest.raises(ValueError, match="eligibility"):
        load_team_pass_correction(
            model.parent,
            parent_lineage_hash="sha256:parent",
            root=root,
            expected_receipt_hash=h,
            depth=0,
            stage="single-mode",
        )
