import copy

import numpy as np
import pytest
import torch

from rosclaw_soccer.rsi.paid_contact_residual import DecoupledContactResidual
from rosclaw_soccer.rsi.paid_full_entry_student import (
    FIRST_R2_CASES,
    FULL_ENTRY_STATUS,
    IDENTITIES,
    full_entry_config,
    validate_full_entry_data,
    validate_full_entry_state,
)


def data_fixture():
    records = []
    for role, cases, skills in (
        ("novel_first_cartesian_teacher", [4, 5], [0]),
        ("canonical_first_R2_teacher", FIRST_R2_CASES, [0]),
        ("original_acquired_success", list(range(16)), [0, 1, 2]),
        ("new_train_acquired_success", list(range(20)), [1, 2]),
    ):
        for skill in skills:
            limit = (
                {0: 16, 1: 14, 2: 15}[skill]
                if role == "original_acquired_success"
                else {1: 19, 2: 16}.get(skill, len(cases))
            )
            for case in cases[:limit]:
                i = len(records)
                records.append(
                    dict(
                        role=role,
                        case=case,
                        skill=skill,
                        rows=1,
                        start=i,
                        stop=i + 1,
                        recorded_policy="base" if "teacher" in role or skill == 0 else "parent",
                    )
                )
    assert len(records) == 94
    n = len(records)
    teacher = np.array(["teacher" in r["role"] for r in records])
    prior = np.full((n, 29), 0.1, np.float32)
    target = np.zeros((n, 29), np.float32)
    target[:2] = 0.05
    target[2:14] = -0.1
    arrays = dict(
        features=np.zeros((n, 288), np.float32),
        target=target,
        parent_delta=prior,
        teacher=teacher,
        group=np.arange(n, dtype=np.int32),
        active=teacher.copy(),
    )
    common = dict(
        source_head="source",
        parent_root="parent",
        parent_hash="hash",
        old_protected_by_kick=[16, 14, 15],
        new_protected_by_kick=[0, 19, 16],
        bank="train",
        Fresh_fit_rows=0,
        DEV_fit_rows=0,
        hardware_authorized=False,
        promotion_authorized=False,
    )
    d = dict(
        **common,
        status="PAID_FULL_ENTRY_TRAIN_FEATURES_CAUSALLY_RECOMPUTED",
        rows=n,
        teacher_rows=14,
        protected_rows=80,
        active_rows=14,
        negative_support_rows=80,
        positive_core_rows=14,
        own_base_feature_rows=n,
        own_parent_head_rows=n,
        trajectories=records,
        optimizer_updates=0,
        own_native_steps=0,
    )
    decl = dict(
        **common,
        schema="paid.full_entry.TRAIN_features.v1",
        teacher_seed_index=0,
        novel_first_teacher_cases=[4, 5],
        canonical_R2_first_cases=FIRST_R2_CASES,
        expected_trajectories=94,
        first_state_has_one_canonical_teacher=True,
        final_R2_relative_bound_rad=0.2,
        internal_difference_bound_rad=0.4,
        runtime_case_lookup=False,
        features="own_normalized160+own_R2_hidden128",
        target="radian_difference_from_frozen_router_parent",
    )
    return d, decl, arrays


def test_complete_causal_full_entry_archive_is_admitted():
    validate_full_entry_data(*data_fixture())


@pytest.mark.parametrize(
    "fault",
    [
        "missing",
        "first_teacher",
        "false_parent",
        "group",
        "active",
        "protected_target",
        "bound",
        "nan",
        "dtype",
        "fresh",
    ],
)
def test_ambiguous_or_unsafe_data_cannot_enter_learning(fault):
    d, decl, z = data_fixture()
    if fault == "missing":
        d["trajectories"].pop()
    elif fault == "first_teacher":
        d["trajectories"][0]["case"] = 14
    elif fault == "false_parent":
        d["trajectories"][2]["recorded_policy"] = "parent"
    elif fault == "group":
        z["group"][2] = 0
    elif fault == "active":
        z["active"][20] = True
    elif fault == "protected_target":
        z["target"][20, 0] = 0.1
    elif fault == "bound":
        z["target"][0, 0] = 0.5
    elif fault == "nan":
        z["features"][0, 0] = np.nan
    elif fault == "dtype":
        z["features"] = z["features"].astype(np.float64)
    else:
        d["Fresh_fit_rows"] = 1
    with pytest.raises(ValueError):
        validate_full_entry_data(d, decl, z)


def state_fixture():
    initial = {k: v.clone() for k, v in DecoupledContactResidual().state_dict().items()}

    def optimizer(prefix, steps):
        values = [v for k, v in initial.items() if k.startswith(prefix)]
        return dict(
            param_groups=[dict(params=list(range(len(values))), lr=0.0003)],
            state={
                i: dict(
                    step=torch.tensor(float(steps)),
                    exp_avg=torch.zeros_like(v),
                    exp_avg_sq=torch.ones_like(v),
                )
                for i, v in enumerate(values)
            },
        )

    counts = dict(
        optimizer_updates=1536,
        gate_optimizer_updates=512,
        head_optimizer_updates=1024,
        positive_forward_rows=524288,
        retention_forward_rows=524288,
    )
    cert = dict(
        maximum_negative_logit_CPU=-0.1, all_protected_predictions_exact_parent=True, batch_shape=1
    )
    calibration = dict(
        maximum_negative_after_CPU=-0.1, only_output_bias_shifted=True, batch_shape=1
    )
    identity = {k: k for k in IDENTITIES}
    state = dict(
        schema="paid.full_entry_student.v1",
        config=full_entry_config(0),
        **identity,
        counts=counts,
        retention_certificate=cert,
        calibration=calibration,
        initial_plastic=initial,
        plastic=copy.deepcopy(initial),
        gate_optimizer=optimizer("gate.", 512),
        optimizer=optimizer("net.", 1024),
        hardware_authorized=False,
        promotion_authorized=False,
    )
    receipt = dict(
        status=FULL_ENTRY_STATUS,
        config=state["config"],
        **identity,
        counts=copy.deepcopy(counts),
        retention_certificate=copy.deepcopy(cert),
        calibration=copy.deepcopy(calibration),
        parent_parameters_in_optimizer=0,
        base_parameters_in_optimizer=0,
        parent_parameters_exact_unchanged=True,
        Fresh_fit_rows=0,
        DEV_fit_rows=0,
        own_native_steps=0,
        hardware_authorized=False,
        promotion_authorized=False,
    )
    return state, receipt, initial


def test_exact_staged_head_and_gate_state_is_admitted():
    validate_full_entry_state(*state_fixture())


def test_v2_changes_support_not_initialization_authority_or_training_budget():
    old = full_entry_config(0)
    new = full_entry_config(0, trajectory_support=True)
    for k in (
        "seed",
        "gate_steps",
        "head_steps",
        "lr",
        "batch_rows",
        "final_bound_rad",
        "internal_difference_bound_rad",
        "parent_frozen",
        "base_frozen",
    ):
        assert old[k] == new[k]
    assert new["protocol"] != old["protocol"] and "trajectory_support" not in old


def test_head_minibatch_stream_is_identical_after_the_support_ablation():
    from rosclaw_soccer.rsi.paid_full_entry_student import head_rng_after_v1_gate

    core = np.arange(188, dtype=np.int64) * 3
    old = np.random.default_rng(2026102800)
    for _ in range(512):
        old.choice(core, 512, replace=True)
        old.integers(0, 21721, size=256)
    new = head_rng_after_v1_gate(2026102800, core, 21721)
    assert np.array_equal(old.choice(3228, 512, replace=True), new.choice(3228, 512, replace=True))
    assert np.array_equal(
        old.choice(18694, 512, replace=True), new.choice(18694, 512, replace=True)
    )


def test_whole_teacher_support_can_open_zero_target_prefix_without_weakening_protected_gate():
    from test_paid_full_entry_contact import model

    from rosclaw_soccer.rsi.paid_full_entry_student import full_entry_certificate

    m = model().eval()
    with torch.no_grad():
        g = m.plastic.gate
        g[0].weight.zero_()
        g[0].bias.zero_()
        g[0].weight[0, 0] = 1
        g[-1].weight.zero_()
        g[-1].weight[0, 0] = 1
        g[-1].bias.fill_(-0.2)
    x = np.zeros((3, 288), np.float32)
    x[:, 0] = [-1, 1, 2]
    with torch.no_grad():
        prior = torch.cat([m.parent(torch.from_numpy(row).reshape(1, 288)) for row in x]).numpy()
    y = np.zeros((3, 29), np.float32)
    y[2] = 0.05
    z = dict(
        features=x,
        parent_delta=prior,
        target=y,
        teacher=np.array([False, True, True]),
        active=np.array([False, False, True]),
    )
    with pytest.raises(ValueError, match="negative-support"):
        full_entry_certificate(m, z)
    cert = full_entry_certificate(m, z, trajectory_support=True)
    assert cert["negative_support_rows"] == 1 and cert["teacher_gate_open_rows"] == 2
    assert cert["protected_rows"] == 1 and cert["all_protected_predictions_exact_parent"]


@pytest.mark.parametrize(
    "fault",
    [
        "init",
        "gate_step",
        "net_step",
        "moments",
        "parent",
        "updates",
        "dtype",
        "nan",
        "calibration",
        "fresh",
        "identity",
    ],
)
def test_partial_or_forged_training_state_is_rejected(fault):
    s, r, initial = state_fixture()
    s = copy.deepcopy(s)
    k = next(iter(initial))
    if fault == "init":
        s["initial_plastic"][k].add_(1)
    elif fault == "gate_step":
        s["gate_optimizer"]["state"][0]["step"] = torch.tensor(511.0)
    elif fault == "net_step":
        s["optimizer"]["state"][0]["step"] = torch.tensor(512.0)
    elif fault == "moments":
        s["optimizer"]["state"][0]["exp_avg_sq"].fill_(-1)
    elif fault == "parent":
        r["parent_parameters_in_optimizer"] = 1
    elif fault == "updates":
        s["counts"]["optimizer_updates"] = 512
    elif fault == "dtype":
        s["plastic"][k] = s["plastic"][k].double()
    elif fault == "nan":
        s["plastic"][k].fill_(float("nan"))
    elif fault == "calibration":
        s["calibration"]["batch_shape"] = 512
    elif fault == "fresh":
        r["Fresh_fit_rows"] = 1
    else:
        s["body_scene_hash"] = "different"
    with pytest.raises(ValueError):
        validate_full_entry_state(s, r, initial)
