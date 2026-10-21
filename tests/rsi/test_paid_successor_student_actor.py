import copy

import pytest
import torch

from rosclaw_soccer.rsi.paid_contact_residual import ContactResidual
from rosclaw_soccer.rsi.paid_successor_features import (
    staged_successor_config,
    successor_student_config,
)
from rosclaw_soccer.rsi.paid_successor_student_actor import validate_successor_state


def fixture(*, staged=False):
    config = staged_successor_config(0) if staged else successor_student_config(0)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(config["seed"])
        template = ContactResidual().state_dict()
    state = dict(
        schema="soccer.successor_contact_student.v1",
        config=config,
        counts=dict(
            optimizer_updates=1024,
            projection_calls=1024,
            projection_gate_evaluations=1024,
            positive_forward_rows=524288,
            retention_forward_rows=524288,
        ),
        projection_histogram={"1.0": 1024},
        retention_certificate=dict(
            maximum_negative_logit_CPU=-0.2, all_negative_predictions_exact_zero=True
        ),
        residual=copy.deepcopy(template),
        initial_residual=copy.deepcopy(template),
        optimizer=dict(
            param_groups=[dict(params=list(range(10)), lr=0.0003)],
            state={
                i: dict(
                    step=torch.tensor(1024.0),
                    exp_avg=torch.zeros_like(v),
                    exp_avg_sq=torch.zeros_like(v),
                )
                for i, v in enumerate(template.values())
            },
        ),
        hardware_authorized=False,
        promotion_authorized=False,
    )
    for key in (
        "source_head",
        "declaration_hash",
        "data_hash",
        "features_hash",
        "base_wrapper_checkpoint_hash",
        "base_wrapper_training_receipt_hash",
        "body_scene_hash",
    ):
        state[key] = "fixture"
    if staged:
        state["schema"] = "soccer.successor_contact_student.v2"
        state["counts"].update(
            optimizer_updates=1536,
            projection_calls=0,
            projection_gate_evaluations=0,
            head_optimizer_updates=1024,
            gate_optimizer_updates=512,
        )
        state["projection_histogram"] = {}
        state["gate_calibration"] = dict(
            all_negative_gate_rejected=True,
            only_output_bias_shifted=True,
            maximum_negative_after_CPU=-0.2,
            margin=0.1,
        )

        def opt(tensors, steps):
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

        state["optimizer"] = opt([v for k, v in template.items() if k.startswith("net.")], 1024)
        state["gate_optimizer"] = opt(
            [v for k, v in template.items() if k.startswith("gate.")], 512
        )
    receipt = dict(
        state,
        status="PAID_SUCCESSOR_NEURAL_STUDENT_TRAINED_NOT_PHYSICALLY_QUALIFIED",
        base_parameters_in_optimizer=0,
        DEV_episodes_fit=0,
        own_native_steps=0,
    )
    return state, receipt, template


def test_complete_new_schema_does_not_require_old_3810_row_protocol():
    validate_successor_state(*fixture())


def test_staged_student_has_two_explicit_optimizers_and_frozen_calibrated_gate():
    validate_successor_state(*fixture(staged=True))


@pytest.mark.parametrize("fault", ["gate_steps", "calibration", "shared_optimizer"])
def test_staged_admission_rejects_missing_independent_stage_work(fault):
    state, receipt, template = fixture(staged=True)
    if fault == "gate_steps":
        state["gate_optimizer"]["state"][0]["step"] = torch.tensor(511.0)
    elif fault == "calibration":
        state["gate_calibration"]["all_negative_gate_rejected"] = False
    else:
        state["optimizer"]["param_groups"][0]["params"] = list(range(10))
    with pytest.raises(ValueError):
        validate_successor_state(state, receipt, template)


@pytest.mark.parametrize(
    "fault", ["schema", "optimizer", "step", "tensor", "base", "DEV", "safety"]
)
def test_rejects_forged_student_admission_or_mutated_base(fault):
    state, receipt, template = fixture()
    if fault == "schema":
        state["schema"] = "soccer.paid_contact_residual.v2"
    elif fault == "optimizer":
        state["optimizer"]["param_groups"][0]["params"].append(10)
    elif fault == "step":
        state["optimizer"]["state"][0]["step"] = torch.tensor(1023.0)
    elif fault == "tensor":
        state["residual"]["gate.2.bias"][0] = float("nan")
    elif fault == "base":
        receipt["base_parameters_in_optimizer"] = 1
    elif fault == "DEV":
        receipt["DEV_episodes_fit"] = 1
    else:
        receipt["hardware_authorized"] = True
    with pytest.raises(ValueError):
        validate_successor_state(state, receipt, template)
