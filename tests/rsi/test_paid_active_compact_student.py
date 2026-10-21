import copy

import pytest
import torch

from rosclaw_soccer.rsi.paid_active_compact_student import (
    ACTIVE_STATUS,
    active_compact_config,
    validate_active_payload,
)
from rosclaw_soccer.rsi.paid_compact_student import FIT_STATUS, compact_config


def fixture():
    source = dict(
        status=FIT_STATUS, config=compact_config(0), data_hash="data", body_scene_hash="body"
    )
    state = dict(
        schema="paid.active_compact_support.v1",
        config=active_compact_config(0),
        data_hash="data",
        body_scene_hash="body",
        source_head="head",
        source_fit_root="source",
        source_fit_hash="hash",
        support=dict(centres=torch.zeros(201, 288), radii=torch.ones(201, dtype=torch.float64)),
        hardware_authorized=False,
        promotion_authorized=False,
    )
    receipt = copy.deepcopy(state)
    receipt.update(
        status=ACTIVE_STATUS,
        optimizer_updates=0,
        inherited_optimizer_updates=1024,
        own_native_steps=0,
        Fresh_fit_rows=0,
        DEV_fit_rows=0,
        head_weights_exact_source=True,
    )
    return state, receipt, source


def test_support_requalification_never_claims_new_neural_training():
    s, r, source = fixture()
    validate_active_payload(s, r, source)
    assert r["optimizer_updates"] == 0
    assert r["config"]["inherited_optimizer_updates"] == 1024


def test_active_support_forces_zero_prefix_without_changing_neural_weights():
    import numpy as np
    from test_paid_compact_student import fixture as motor_fixture

    from rosclaw_soccer.rsi.paid_compact_student import audit_compact_batch1
    from rosclaw_soccer.rsi.paid_compact_support import CompactTeacherSupport

    head, positive, negative, _, receipt, _ = motor_fixture()
    before = {k: v.clone() for k, v in head.net.state_dict().items()}
    head.support = CompactTeacherSupport(positive[:1], torch.full((1,), 0.45, dtype=torch.float64))
    features = torch.cat([positive, negative]).numpy()
    active = np.array([True, False, False, False, False])
    z = dict(
        features=features,
        parent_delta=np.zeros((5, 29), np.float32),
        target=np.zeros((5, 29), np.float32),
        teacher=np.array([True, True, False, False, False]),
        active=active,
    )
    z["target"][0] = 0.1
    with torch.no_grad():
        mse = float((head.combine(positive[:1], torch.zeros(1, 29)) - 0.1).square().mean())
    receipt["retention_certificate"].update(
        protected_rows_exact_parent=3,
        teacher_centres_gain_exact_one=2,
        active_correction_MSE=mse,
        no_op_active_correction_MSE=0.01,
    )
    cert = audit_compact_batch1(head, receipt, z, active_only=True)
    assert cert["teacher_centres_gain_exact_one"] == 1
    assert cert["inactive_teacher_rows_exact_parent"] == 1
    assert torch.equal(head.combine(positive[1:], torch.zeros(1, 29)), torch.zeros(1, 29))
    assert all(torch.equal(v, before[k]) for k, v in head.net.state_dict().items())


@pytest.mark.parametrize(
    "fault",
    [
        "source",
        "nested",
        "updates",
        "radius",
        "nan",
        "dtype",
        "body",
        "weights",
        "authority",
        "centres",
    ],
)
def test_source_requalification_has_no_recursive_head_or_authority_bypass(fault):
    s, r, source = fixture()
    if fault == "source":
        source["status"] = ACTIVE_STATUS
    elif fault == "nested":
        source["config"] = active_compact_config(0)
    elif fault == "updates":
        r["optimizer_updates"] = 1024
    elif fault == "radius":
        s["support"]["radii"].fill_(0)
    elif fault == "nan":
        s["support"]["centres"][0, 0] = float("nan")
    elif fault == "dtype":
        s["support"]["centres"] = s["support"]["centres"].double()
    elif fault == "body":
        source["body_scene_hash"] = "other"
    elif fault == "weights":
        r["head_weights_exact_source"] = False
    elif fault == "authority":
        r["hardware_authorized"] = True
    else:
        s["support"]["centres"] = torch.zeros(3228, 288)
    with pytest.raises(ValueError):
        validate_active_payload(s, r, source)


def test_dirty_support_ablation_cannot_load_source_or_create_evidence(tmp_path, monkeypatch):
    import sys

    from scripts import rsi_requalify_paid_active_support as runner

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run",
            "--source-root",
            str(tmp_path / "source"),
            "--source-hash",
            "hash",
            "--expected-head",
            "head",
            "--output",
            str(tmp_path / "out"),
        ],
    )
    monkeypatch.setattr("scripts.rsi_native_context_recovery_common.digest", lambda _: "hash")
    monkeypatch.setattr(runner.subprocess, "check_output", lambda *a, **k: b"dirty")
    with pytest.raises(ValueError, match="frozen"):
        runner.main()
    assert not (tmp_path / "out").exists()
