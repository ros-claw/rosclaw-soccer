import copy

import numpy as np
import pytest
import torch
from test_paid_compact_student import fixture as warm_fixture

from rosclaw_soccer.rsi.paid_interventional_student import (
    CONTROL_STATUS,
    STUDENT_STATUS,
    learner_config,
    learner_status,
    learning_certificate,
    precision_imitation_loss,
    validate_learner_state,
)


def fixture(control=False, generation=1):
    head, p, n, warm, _, _ = warm_fixture()
    initial = {
        k.removeprefix("net."): v.clone() for k, v in warm["model"].items() if k.startswith("net.")
    }
    if not control:
        with torch.no_grad():
            head.net[-1].bias.add_(0.001)
    state = dict(
        schema=f"paid.interventional_dagger_student.v{generation}",
        config=learner_config(0, control=control, generation=generation),
        head=copy.deepcopy(head.state_dict()),
        initial_net=initial,
        optimizer=copy.deepcopy(warm["optimizer"]),
        hardware_authorized=False,
        promotion_authorized=False,
    )
    steps = 0 if control else 2048
    inherited = {1: 1024, 2: 3072, 3: 5120}[generation]
    for slot in state["optimizer"]["state"].values():
        slot["step"].fill_(inherited + steps)
    state["counts"] = dict(
        optimizer_updates=steps,
        inherited_optimizer_updates=inherited,
        canonical_forward_rows=256 * steps,
        intervention_forward_rows=256 * steps,
        retention_forward_rows=512 * steps,
        nonzero_gradient_steps=steps,
    )
    for k in (
        "source_head",
        "declaration_hash",
        "data_hash",
        "warm_hash",
        "body_scene_hash",
        "parent_hash",
        "parent_checkpoint_hash",
    ):
        state[k] = k
    z = dict(
        features=torch.cat([p, n]).numpy(),
        target=np.ones((5, 29), np.float32) * 0.1,
        parent_delta=np.zeros((5, 29), np.float32),
        teacher=np.array([True, True, False, False, False]),
        active=np.array([True, True, False, False, False]),
    )
    state["learning_certificate"] = learning_certificate(head, z, 1, initial, control=control)
    receipt = copy.deepcopy(state)
    receipt.update(
        status=learner_status(generation, control),
        parent_parameters_in_optimizer=0,
        base_parameters_in_optimizer=0,
        own_native_steps=0,
        Fresh_fit_rows=0,
        DEV_fit_rows=0,
    )
    return state, receipt, warm


@pytest.mark.parametrize("control", [False, True])
def test_replanned_learning_keeps_3072_warm_updates_separate_from_new_steps(control):
    s, r, w = fixture(control, generation=2)
    validate_learner_state(s, r, w)
    assert s["counts"]["inherited_optimizer_updates"] == 3072
    assert s["config"]["Adam_total_steps"] == (3072 if control else 5120)
    assert s["config"]["seed"] == 2026110100
    s["counts"]["inherited_optimizer_updates"] = 1024
    with pytest.raises(ValueError):
        validate_learner_state(s, r, w)


@pytest.mark.parametrize("generation", [True, 0, 4, "2"])
def test_learning_generation_is_typed_and_bounded(generation):
    with pytest.raises(ValueError, match="generation"):
        learner_config(0, generation=generation)


@pytest.mark.parametrize("control", [False, True])
def test_precision_refinement_inherits_all_5120_actual_updates(control):
    s, r, w = fixture(control, generation=3)
    validate_learner_state(s, r, w)
    assert s["config"]["teacher_loss"] == "unclipped_relative_correction_mse"
    assert s["config"]["Adam_total_steps"] == (5120 if control else 7168)
    assert s["counts"]["inherited_optimizer_updates"] == 5120
    s["schema"] = "paid.interventional_dagger_student.v2"
    with pytest.raises(ValueError):
        validate_learner_state(s, r, w)


def test_precision_loss_has_gradient_where_runtime_clamp_has_none():
    raw = torch.full((2, 29), 0.3, requires_grad=True)
    target = torch.full_like(raw, 0.1)
    old = (raw.clamp(-0.2, 0.2) - target).square().mean()
    old.backward()
    assert raw.grad.count_nonzero() == 0
    raw.grad.zero_()
    loss = precision_imitation_loss(raw, target, torch.ones(2, dtype=torch.bool))
    loss.backward()
    assert (raw.grad > 0).all()
    assert learner_config(0, generation=3)["relative_scale_floor_rad"] == 0.02


def test_precision_loss_protects_small_onset_from_large_action_average():
    active = torch.ones(1, dtype=torch.bool)
    quiet = torch.zeros(1, 29)
    large = torch.full((1, 29), 0.2)
    early = precision_imitation_loss(quiet + 0.001, quiet, active)
    later = precision_imitation_loss(large + 0.001, large, active)
    assert float(early / later) == pytest.approx(100, rel=1e-4)
    with pytest.raises(ValueError, match="finite"):
        precision_imitation_loss(quiet.fill_(float("nan")), large, active)


@pytest.mark.parametrize("control", [False, True])
def test_new_gradients_and_zero_update_control_have_distinct_contracts(control):
    s, r, w = fixture(control)
    validate_learner_state(s, r, w)
    assert r["learning_certificate"]["new_neural_weight_effect"] is (not control)
    assert r["learning_certificate"]["counts"]["protected"] == 3
    assert r["counts"]["inherited_optimizer_updates"] == 1024


@pytest.mark.parametrize(
    "fault", ["steps", "source", "noop", "tensor", "initial", "count", "flags", "moment"]
)
def test_no_forged_updates_or_lost_warm_memory(fault):
    s, r, w = fixture()
    if fault == "steps":
        s["optimizer"]["state"][0]["step"].fill_(1024)
    elif fault == "source":
        r["warm_hash"] = "other"
    elif fault == "noop":
        for k, v in s["initial_net"].items():
            s["head"]["net." + k] = v.clone()
    elif fault == "tensor":
        s["head"]["net.4.bias"][0] = float("nan")
    elif fault == "initial":
        s["initial_net"]["4.bias"].fill_(42)
    elif fault == "count":
        s["counts"]["inherited_optimizer_updates"] = 0
    elif fault == "flags":
        r["promotion_authorized"] = True
    elif fault == "moment":
        s["optimizer"]["state"][0]["exp_avg_sq"].fill_(-1)
    with pytest.raises(ValueError):
        validate_learner_state(s, r, w)


def test_control_cannot_masquerade_as_neural_learning_or_change_moments():
    s, r, w = fixture(True)
    r["status"] = STUDENT_STATUS
    with pytest.raises(ValueError):
        validate_learner_state(s, r, w)
    r["status"] = CONTROL_STATUS
    s["optimizer"]["state"][0]["exp_avg"].add_(1)
    with pytest.raises(ValueError, match="control"):
        validate_learner_state(s, r, w)


@pytest.mark.parametrize("index", [-1, 3, True, "0"])
def test_seed_and_causal_control_design_is_fixed(index):
    with pytest.raises(ValueError):
        learner_config(index)
    with pytest.raises(ValueError):
        learner_config(1, control=True)


def test_dirty_learner_rejects_before_source_reads_or_GPU_allocation(tmp_path, monkeypatch):
    import sys

    from scripts import rsi_train_paid_interventional as runner

    args = ["run", "--expected-head", "frozen", "--seed", "0", "--output", str(tmp_path / "out")]
    for name in ("data", "warm"):
        args += ["--" + name + "-root", str(tmp_path / name), "--" + name + "-hash", "hash"]
    monkeypatch.setattr(sys, "argv", args)
    monkeypatch.setattr(runner.subprocess, "check_output", lambda *a, **k: b"dirty")
    with pytest.raises(ValueError, match="frozen"):
        runner.main()
    assert not (tmp_path / "out").exists()
