"""Synthetic admission fixtures; never presented as actual physical or fit evidence."""

import json

import pytest
import torch
from test_consolidated_adapter_learning import synthetic_checkpoint
from test_consolidated_motor_adapter import learned_memory
from test_single_mode_pass_curriculum import records

import rosclaw_soccer.rsi.consolidated_adapter_checkpoint as admission
from rosclaw_soccer.rsi.consolidated_adapter_learning import (
    FEATURE_STATUS,
    KIND,
    STATUS,
    adapter_fit_plan,
    adapter_schedule,
)
from rosclaw_soccer.rsi.consolidated_motor_adapter import ConsolidatedMotorAdapter
from rosclaw_soccer.rsi.dual_motor_checkpoint import digest


def fixture(tmp_path, monkeypatch, fault=None):
    def write(folder, value):
        folder.mkdir()
        p = folder / "complete.json"
        p.write_text(json.dumps(value))
        return p, digest(p)

    rows = records()
    for row in rows:
        row["rows"] = 360
    state, _, cfg, _ = synthetic_checkpoint()
    parent_hash = "sha256:parent"
    bp, bh = write(tmp_path / "bank", dict(records=rows, parent_lineage_hash=parent_hash))
    mp, mh = write(
        tmp_path / "memory",
        dict(
            checkpoint_hash="sha256:whole_memory",
            cumulative_optimizer_steps=112,
            cumulative_critic_optimizer_steps=48,
        ),
    )
    fp, fh = write(
        tmp_path / "features",
        dict(
            status=FEATURE_STATUS,
            data_hash=bh,
            memory_training_hash=mh,
            memory_checkpoint_hash="sha256:whole_memory",
            zero_head_bit_identity=True,
            teacher_hidden_state_supplied=False,
            records=[{}] * 274,
            own_feature_decisions=274 * 360,
        ),
    )
    plan = adapter_fit_plan(rows)
    ids = dict(
        feature_hash=fh,
        data_hash=bh,
        fit_plan_hash=plan["hash"],
        memory_training_hash=mh,
        memory_checkpoint_hash="sha256:whole_memory",
        parent_lineage_hash=parent_hash,
    )
    state.update(ids)
    root = tmp_path / "candidate"
    root.mkdir()
    cp = root / "head.pt"
    if fault == "head":
        state["head"]["2.bias"].fill_(float("nan"))
    if fault == "old_counter":
        state["head_optimizer"]["state"][0]["step"].fill_(112)
    torch.save(state, cp)
    pins = {str(p): h for p, h in ((bp, bh), (mp, mh), (fp, fh))}
    d = dict(
        source_head="fixture",
        config=cfg,
        input_file_hashes=pins,
        fit_plan=plan,
        **ids,
        goal_complete=False,
        promotion_authorized=False,
        hardware_authorized=False,
    )
    dp = root / "declaration.json"
    dp.write_text(json.dumps(d))
    history = [
        entry | dict(loss=0.1, imitation=0.1, retention=0.0, baseline=0.0, gradient_norm=0.01)
        for entry in adapter_schedule(plan, cfg["seed"])
    ]
    r = dict(
        d,
        status=STATUS,
        checkpoint_kind=KIND,
        declaration_hash=digest(dp),
        head_optimizer_steps=2048,
        memory_optimizer_steps_unchanged=dict(actor=112, critic=48),
        physical_steps_executed=0,
        old_retention_proven=False,
        online_RL_claim=False,
        history=history,
        checkpoint_path=str(cp),
        checkpoint_hash=digest(cp),
    )
    if fault == "schedule":
        r["history"][0]["teacher_indices"][0] = 0
    elif fault == "plan":
        r["fit_plan"]["old_anchor_indices"] = list(range(177, 194))
    elif fault == "promotion":
        r["promotion_authorized"] = True
    elif fault == "ancestry":
        r["memory_checkpoint_hash"] = "sha256:other"
    elif fault == "escape":
        r["checkpoint_path"] = str(tmp_path / "outside.pt")
    rp = root / "complete.json"
    rp.write_text(json.dumps(r))
    memory = learned_memory()
    monkeypatch.setattr(
        admission, "load_dual_candidate_for_sim_audit", lambda *args, **kwargs: (memory, {})
    )
    return root, digest(rp), memory


def test_loader_admits_only_explicit_new_head_without_reinterpreting_old_motor(
    tmp_path, monkeypatch
):
    root, h, memory = fixture(tmp_path, monkeypatch)
    actor, r = admission.load_motor_candidate_for_sim_audit(
        memory.parent,
        parent_lineage_hash="sha256:parent",
        root=root,
        expected_receipt_hash=h,
        checkpoint_kind=KIND,
    )
    assert type(actor) is ConsolidatedMotorAdapter
    assert not actor.training and not any(p.requires_grad for p in actor.parameters())
    assert r["memory_optimizer_steps_unchanged"] == dict(actor=112, critic=48)
    assert all(torch.equal(v, memory.state_dict()[k]) for k, v in actor.memory.state_dict().items())


@pytest.mark.parametrize(
    "fault", ["head", "old_counter", "schedule", "plan", "promotion", "ancestry", "escape"]
)
def test_loader_rejects_changed_lineage_scope_or_fabricated_training(tmp_path, monkeypatch, fault):
    root, h, memory = fixture(tmp_path, monkeypatch, fault)
    with pytest.raises(ValueError):
        admission.load_consolidated_adapter(
            memory.parent, parent_lineage_hash="sha256:parent", root=root, expected_receipt_hash=h
        )


def test_legacy_dispatch_does_not_reinterpret_original_checkpoint_kind(monkeypatch, tmp_path):
    memory = learned_memory()
    seen = []

    def old_loader(*args, **kwargs):
        seen.append(kwargs)
        return memory, {}

    monkeypatch.setattr(admission, "load_dual_candidate_for_sim_audit", old_loader)
    actor, _ = admission.load_motor_candidate_for_sim_audit(
        memory.parent,
        parent_lineage_hash="parent",
        root=tmp_path,
        expected_receipt_hash="receipt",
        checkpoint_kind="team-pass-selected-v3",
    )
    assert actor is memory and seen[0]["checkpoint_kind"] == "team-pass-selected-v3"


def test_refinement_data_identity_cannot_be_aliased_to_legacy_bank():
    kind = "consolidated-refinement-v1"
    bank = "sha256:" + "a" * 64
    assert admission.training_data_hash(dict(bank_hash=bank), kind) == bank
    with pytest.raises(ValueError, match="contradictory"):
        admission.training_data_hash(dict(bank_hash=bank, data_hash="old274"), kind)
    with pytest.raises(ValueError, match="commitment"):
        admission.training_data_hash(dict(bank_hash="short"), kind)


def test_refinement_cpu_scope_comes_from_hash_bound_whole_feature_witness(tmp_path):
    from rosclaw_soccer.rsi.refinement_motor_learning import FEATURE_STATUS

    root = tmp_path / "features"
    root.mkdir()
    path = root / "complete.json"
    feature = dict(
        status=FEATURE_STATUS,
        bank_hash="bank62",
        own_feature_decisions=23241,
        records=[dict(rows=375)] * 61 + [dict(rows=23241 - 375 * 61)],
    )
    path.write_text(json.dumps(feature))
    training = dict(
        bank_hash="bank62",
        feature_hash=digest(path),
        input_file_hashes={str(path): digest(path)},
    )
    assert admission.expected_cpu_audit_decisions(training, "consolidated-refinement-v1") == 23241
    training["bank_hash"] = "old274"
    with pytest.raises(ValueError, match="scope"):
        admission.expected_cpu_audit_decisions(training, "consolidated-refinement-v1")
