import copy

import numpy as np
import pytest
import torch

from rosclaw_soccer.rsi.paid_contact_residual import DecoupledContactResidual
from rosclaw_soccer.rsi.paid_progressive_student import (
    DATA_STATUS,
    TRAINED_STATUS,
    progressive_config,
    validate_progressive_arrays,
    validate_progressive_data,
    validate_progressive_state,
)


def data_fixture():
    declaration = dict(
        schema="soccer.progressive_contact_curriculum.v1",
        teacher_cases=[8, 9],
        final_R2_relative_bound_radians=0.2,
        internal_difference_bound_radians=0.4,
        features="own_normalized160+own_causal128",
        runtime_case_lookup=False,
        bank="train",
        DEV_episodes_fit=0,
        protected_successes_by_kick=[16, 14, 13],
        parent_hash="parent",
        source_head="source",
        hardware_authorized=False,
        promotion_authorized=False,
    )
    rows = []
    for role, case, skill in [
        ("teacher", 8, 2),
        ("teacher", 9, 2),
        *[
            ("protected_acquired_success", i, k)
            for k, count in enumerate([16, 14, 13])
            for i in range(count)
        ],
    ]:
        start = len(rows)
        rows.append(dict(role=role, case=case, skill=skill, rows=1, start=start, stop=start + 1))
    data = dict(
        declaration,
        status=DATA_STATUS,
        trajectories=rows,
        rows=45,
        CPU_base_feature_recompute_rows=45,
        CPU_parent_head_recompute_rows=45,
        teacher_rows=2,
        protected_acquired_rows=43,
        active_rows=2,
        negative_support_rows=43,
        own_native_steps=0,
        optimizer_updates=0,
    )
    teacher = np.zeros(45, np.bool_)
    teacher[:2] = True
    target = np.zeros((45, 29), np.float32)
    target[:2] = 0.1
    arrays = dict(
        features=np.zeros((45, 288), np.float32),
        target=target,
        parent_delta=np.full((45, 29), 0.02, np.float32),
        teacher=teacher,
        active=teacher.copy(),
        group=np.arange(45, dtype=np.int32),
    )
    return data, declaration, arrays


def state_fixture():
    config = progressive_config(0)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(config["seed"])
        template = DecoupledContactResidual().state_dict()

    def opt(prefix, steps):
        tensors = [v for k, v in template.items() if k.startswith(prefix)]
        return dict(
            param_groups=[dict(params=list(range(len(tensors))), lr=0.0003)],
            state={
                i: dict(
                    step=torch.tensor(float(steps)),
                    exp_avg=torch.zeros_like(v),
                    exp_avg_sq=torch.zeros_like(v),
                )
                for i, v in enumerate(tensors)
            },
        )

    state = dict(
        schema="soccer.progressive_contact_student.v1",
        config=config,
        counts=dict(
            optimizer_updates=1536,
            gate_optimizer_updates=512,
            head_optimizer_updates=1024,
            positive_forward_rows=524288,
            retention_forward_rows=524288,
        ),
        retention_certificate=dict(
            maximum_negative_logit_CPU=-0.2,
            all_negative_predictions_exact_parent=True,
            all_negative_corrections_exact_zero=True,
        ),
        gate_calibration=dict(
            maximum_negative_after_CPU=-0.2,
            only_output_bias_shifted=True,
            all_negative_gate_rejected=True,
        ),
        plastic=copy.deepcopy(template),
        initial_plastic=copy.deepcopy(template),
        optimizer=opt("net.", 1024),
        gate_optimizer=opt("gate.", 512),
        hardware_authorized=False,
        promotion_authorized=False,
    )
    for k in (
        "source_head",
        "declaration_hash",
        "data_hash",
        "features_hash",
        "parent_hash",
        "parent_checkpoint_hash",
        "body_scene_hash",
        "base_wrapper_checkpoint_hash",
        "base_wrapper_training_receipt_hash",
    ):
        state[k] = "fixture"
    receipt = dict(
        copy.deepcopy(state),
        status=TRAINED_STATUS,
        parent_parameters_in_optimizer=0,
        base_parameters_in_optimizer=0,
        parent_parameters_exact_unchanged=True,
        DEV_episodes_fit=0,
        own_native_steps=0,
    )
    return state, receipt, template


def test_parent_relative_archive_is_accepted_with_nonzero_acquired_actions():
    data, declaration, arrays = data_fixture()
    validate_progressive_data(data, declaration)
    validate_progressive_arrays(data, arrays)
    assert arrays["parent_delta"][2:].min() > 0


@pytest.mark.parametrize(
    "fault",
    ["dev", "missing", "duplicate", "range", "memory_count", "parent", "authority", "teacher"],
)
def test_incomplete_or_changed_acquired_curriculum_rejected(fault):
    data, declaration, _ = data_fixture()
    if fault == "dev":
        declaration["bank"] = "dev"
    elif fault == "missing":
        data["trajectories"].pop()
    elif fault == "duplicate":
        data["trajectories"][3]["case"] = 0
    elif fault == "range":
        data["trajectories"][0]["stop"] = 2
    elif fault == "memory_count":
        data["CPU_parent_head_recompute_rows"] -= 1
    elif fault == "parent":
        declaration["parent_hash"] = "other"
    elif fault == "authority":
        declaration["final_R2_relative_bound_radians"] = 0.4
    else:
        data["trajectories"][0]["case"] = 3
    with pytest.raises(ValueError):
        validate_progressive_data(data, declaration)


@pytest.mark.parametrize(
    "fault",
    ["dtype", "nan", "false_zero", "group", "role", "active", "total_authority", "difference"],
)
def test_false_parent_labels_or_fabricated_feature_roles_rejected(fault):
    data, _, arrays = data_fixture()
    if fault == "dtype":
        arrays["features"] = arrays["features"].astype(np.float64)
    elif fault == "nan":
        arrays["features"][0, 0] = np.nan
    elif fault == "false_zero":
        arrays["target"][2, 0] = 0.1
    elif fault == "group":
        arrays["group"][1] = 0
    elif fault == "role":
        arrays["teacher"][2] = True
    elif fault == "active":
        arrays["active"][0] = False
    elif fault == "total_authority":
        arrays["target"][0, 0] = 0.19
    else:
        arrays["target"][0, 0] = -0.41
    with pytest.raises(ValueError):
        validate_progressive_arrays(data, arrays)


def test_complete_progressive_checkpoint_contract():
    state, receipt, template = state_fixture()
    validate_progressive_state(state, receipt, template)


@pytest.mark.parametrize(
    "fault",
    [
        "schema",
        "seed",
        "parent",
        "parent_changed",
        "old_zero_claim",
        "init",
        "nan",
        "steps",
        "extra_optimizer",
        "moment",
        "hardware",
        "partial",
    ],
)
def test_checkpoint_cannot_bypass_parent_authority_or_retention(fault):
    state, receipt, template = state_fixture()
    if fault == "schema":
        state["schema"] = "soccer.successor_contact_student.v3"
    elif fault == "seed":
        state["config"]["seed_index"] = True
    elif fault == "parent":
        receipt["parent_hash"] = "other"
    elif fault == "parent_changed":
        receipt["parent_parameters_exact_unchanged"] = False
    elif fault == "old_zero_claim":
        state["retention_certificate"] = dict(
            maximum_negative_logit_CPU=-0.2, all_negative_predictions_exact_zero=True
        )
    elif fault == "init":
        state["initial_plastic"]["net.0.weight"][0, 0] += 1
    elif fault == "nan":
        state["plastic"]["net.0.weight"][0, 0] = float("nan")
    elif fault == "steps":
        state["optimizer"]["state"][0]["step"] = torch.tensor(1023.0)
    elif fault == "extra_optimizer":
        state["optimizer"]["param_groups"][0]["params"].append(6)
    elif fault == "moment":
        state["gate_optimizer"]["state"][0]["exp_avg_sq"].fill_(-1)
    elif fault == "hardware":
        receipt["hardware_authorized"] = True
    else:
        state["counts"]["head_optimizer_updates"] = 1023
    with pytest.raises(ValueError):
        validate_progressive_state(state, receipt, template)
