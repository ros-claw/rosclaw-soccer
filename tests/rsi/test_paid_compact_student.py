import copy

import numpy as np
import pytest
import torch

from rosclaw_soccer.rsi.paid_compact_student import (
    FIT_STATUS,
    audit_compact_batch1,
    compact_config,
    validate_compact_state,
)
from rosclaw_soccer.rsi.paid_compact_support import (
    CompactSupportedHead,
    CompactTeacherSupport,
    nearest_protected_radii,
)


def fixture():
    p, n = torch.zeros(2, 288), torch.ones(3, 288)
    p[1, 0] = 1
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(2026102800)
        model = CompactSupportedHead(CompactTeacherSupport(p, nearest_protected_radii(p, n)))
    initial = {k: v.clone() for k, v in model.net.state_dict().items()}
    opt = torch.optim.Adam(model.net.parameters(), lr=0.0003)
    model.combine(p, torch.zeros(2, 29)).sub(0.1).square().mean().backward()
    opt.step()
    state = copy.deepcopy(opt.state_dict())
    # Unit contract fixture only, not a claimed training receipt.
    for slot in state["state"].values():
        slot["step"].fill_(1024)
    changed = max(float((v - initial[k]).abs().max()) for k, v in model.net.state_dict().items())
    cert = dict(
        maximum_head_parameter_change=changed,
        nonzero_gradient_steps=1024,
        maximum_protected_gain_CPU=0.0,
        runtime_batch1_not_yet_certified=True,
        physical_qualification=False,
    )
    s = dict(
        schema="paid.compact_support.fit_only.v1",
        config=compact_config(0),
        counts=dict(
            optimizer_updates=1024, positive_forward_rows=524288, retention_forward_rows=524288
        ),
        model=copy.deepcopy(model.state_dict()),
        initial_net=initial,
        optimizer=state,
        retention_certificate=cert,
        hardware_authorized=False,
        promotion_authorized=False,
    )
    for k in (
        "source_head",
        "declaration_hash",
        "data_hash",
        "feature_hash",
        "parent_hash",
        "parent_checkpoint_hash",
        "body_scene_hash",
    ):
        s[k] = k
    r = copy.deepcopy(s)
    r.update(
        status=FIT_STATUS,
        parent_parameters_in_optimizer=0,
        base_parameters_in_optimizer=0,
        Fresh_fit_rows=0,
        DEV_fit_rows=0,
        own_native_steps=0,
    )
    return model, p, n, s, r, initial


def test_effective_fit_contract_and_three_fixed_seeds():
    _, _, _, s, r, initial = fixture()
    validate_compact_state(s, r, initial)
    assert [compact_config(i)["seed"] for i in range(3)] == [2026102800, 2026102801, 2026102802]
    for seed in (-1, 3, True, "0"):
        with pytest.raises(ValueError):
            compact_config(seed)


@pytest.mark.parametrize(
    "mutation",
    [
        "no_op",
        "seed",
        "nan",
        "optimizer",
        "moment",
        "initial",
        "retention",
        "authority",
        "identity",
        "count",
    ],
)
def test_forged_or_invalid_effective_learning_contract_rejected(mutation):
    _, _, _, s, r, initial = fixture()
    if mutation == "no_op":
        for k, v in initial.items():
            s["model"]["net." + k] = v.clone()
    elif mutation == "seed":
        s["config"]["seed"] += 1
    elif mutation == "nan":
        s["model"]["net.4.bias"][0] = float("nan")
    elif mutation == "optimizer":
        s["optimizer"]["state"][0]["step"].fill_(1)
    elif mutation == "moment":
        s["optimizer"]["state"][0]["exp_avg_sq"].fill_(-1)
    elif mutation == "initial":
        s["initial_net"]["4.bias"].fill_(1)
    elif mutation == "retention":
        s["retention_certificate"]["maximum_protected_gain_CPU"] = 0.01
        r["retention_certificate"] = copy.deepcopy(s["retention_certificate"])
    elif mutation == "authority":
        s["promotion_authorized"] = True
    elif mutation == "identity":
        r["parent_hash"] = "different"
    elif mutation == "count":
        s["counts"]["optimizer_updates"] = 1536
    with pytest.raises(ValueError):
        validate_compact_state(s, r, initial)


def test_batch1_independent_readback_and_unchanged_anchors():
    m, p, n, _, r, _ = fixture()
    z = dict(
        features=torch.cat([p, n]).numpy(),
        parent_delta=np.zeros((5, 29), np.float32),
        target=np.full((5, 29), 0.1, np.float32),
        teacher=np.array([True, True, False, False, False]),
        active=np.array([True, True, False, False, False]),
    )
    with torch.no_grad():
        mse = float((m.combine(p, torch.zeros(2, 29)) - 0.1).square().mean())
    r["retention_certificate"].update(
        protected_rows_exact_parent=3,
        teacher_centres_gain_exact_one=2,
        active_correction_MSE=mse,
        no_op_active_correction_MSE=0.01,
    )
    c = audit_compact_batch1(m, r, z)
    assert c["protected_rows_exact_parent"] == 3
    assert c["teacher_centres_gain_exact_one"] == 2
    assert c["batch_shape"] == 1
    assert not c["realtime_guaranteed"] and not c["closed_loop_physics_qualified"]
    r["retention_certificate"]["protected_rows_exact_parent"] = 2
    with pytest.raises(ValueError):
        audit_compact_batch1(m, r, z)


def test_dirty_verification_rejected_before_output(tmp_path, monkeypatch):
    import sys

    from scripts import rsi_verify_paid_compact_support as runner

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run",
            "--fit-root",
            str(tmp_path / "fit"),
            "--fit-hash",
            "hash",
            "--expected-head",
            "frozen",
            "--output",
            str(tmp_path / "out"),
        ],
    )
    monkeypatch.setattr("scripts.rsi_native_context_recovery_common.digest", lambda _: "hash")
    monkeypatch.setattr(runner.subprocess, "check_output", lambda *a, **k: b"dirty")
    with pytest.raises(ValueError, match="frozen"):
        runner.main()
    assert not (tmp_path / "out").exists()
