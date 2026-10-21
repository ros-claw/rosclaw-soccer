import copy

import numpy as np
import pytest
import torch

from rosclaw_soccer.rsi.paid_ancestor_router import AncestralResidualRouter
from rosclaw_soccer.rsi.paid_ancestor_router_student import (
    ROUTER_STATUS,
    calibrate_router_gate,
    router_certificate,
    router_config,
    validate_router_state,
)
from rosclaw_soccer.rsi.paid_contact_residual import DecoupledContactResidual
from rosclaw_soccer.rsi.paid_progressive_contact import ProgressiveContactResidual


def model():
    return AncestralResidualRouter(
        ProgressiveContactResidual(DecoupledContactResidual(), DecoupledContactResidual())
    ).eval()


def valid_state():
    m = model()
    initial = {k: v.clone() for k, v in m.gate.state_dict().items()}
    slots = {
        i: dict(
            step=torch.tensor(512.0), exp_avg=torch.zeros_like(v), exp_avg_sq=torch.ones_like(v)
        )
        for i, v in enumerate(initial.values())
    }
    cert = dict(
        maximum_meaningful_negative_logit_CPU=-0.1,
        all_protected_predictions_exact_parent=True,
        batch_shape=1,
    )
    calibration = dict(
        maximum_negative_after_CPU=-0.1, batch_shape=1, only_output_bias_shifted=True
    )
    counts = dict(optimizer_updates=512, positive_rows=262144, negative_rows=262144)
    identity = {
        k: k
        for k in (
            "source_head",
            "declaration_hash",
            "data_hash",
            "feature_hash",
            "parent_hash",
            "parent_checkpoint_hash",
            "body_scene_hash",
        )
    }
    state = dict(
        schema="paid.ancestral_router_student.v1",
        config=router_config(0),
        gate=copy.deepcopy(initial),
        initial_gate=initial,
        optimizer=dict(state=slots, param_groups=[dict(params=[0, 1, 2, 3], lr=0.0003)]),
        retention_certificate=cert,
        calibration=calibration,
        counts=counts,
        hardware_authorized=False,
        promotion_authorized=False,
        **identity,
    )
    receipt = dict(
        status=ROUTER_STATUS,
        config=state["config"],
        counts=counts,
        retention_certificate=cert,
        calibration=calibration,
        parent_parameters_in_optimizer=0,
        base_parameters_in_optimizer=0,
        parent_parameters_exact_unchanged=True,
        Fresh_fit_rows=0,
        own_native_steps=0,
        hardware_authorized=False,
        promotion_authorized=False,
        **identity,
    )
    return state, receipt, initial


def test_complete_gate_only_training_state_is_admitted():
    validate_router_state(*valid_state())


@pytest.mark.parametrize(
    "fault",
    [
        "updates",
        "step",
        "moments",
        "dtype",
        "nan",
        "initial",
        "keys",
        "base",
        "parent",
        "fresh",
        "body",
        "calibration",
    ],
)
def test_bad_training_contract_is_rejected(fault):
    state, receipt, initial = valid_state()
    # Break aliasing so receipt/state mismatches cannot conceal malformed values.
    state, receipt = copy.deepcopy(state), copy.deepcopy(receipt)
    key = next(iter(initial))
    if fault == "updates":
        state["counts"]["optimizer_updates"] = 511
    elif fault == "step":
        state["optimizer"]["state"][0]["step"] = torch.tensor(511.0)
    elif fault == "moments":
        state["optimizer"]["state"][0]["exp_avg_sq"].fill_(-1)
    elif fault == "dtype":
        state["gate"][key] = state["gate"][key].double()
    elif fault == "nan":
        state["gate"][key].fill_(float("nan"))
    elif fault == "initial":
        state["initial_gate"][key].add_(1)
    elif fault == "keys":
        state["gate"]["unexpected"] = torch.zeros(1)
    elif fault == "base":
        receipt["base_parameters_in_optimizer"] = 1
    elif fault == "parent":
        receipt["parent_parameters_exact_unchanged"] = False
    elif fault == "fresh":
        receipt["Fresh_fit_rows"] = 1
    elif fault == "body":
        receipt["body_scene_hash"] = "different"
    else:
        state["calibration"]["batch_shape"] = 64
    with pytest.raises(ValueError):
        validate_router_state(state, receipt, initial)


def test_cpu_calibration_changes_only_output_bias_and_rejects_every_negative():
    m = model()
    with torch.no_grad():
        m.gate[-1].weight.fill_(0.03)
        m.gate[-1].bias.fill_(2)
    before = {k: v.clone() for k, v in m.gate.state_dict().items()}
    x = torch.randn(19, 288)
    result = calibrate_router_gate(m.gate, x)
    assert result["CPU_gate_rows"] == 38 and result["batch_shape"] == 1
    assert result["maximum_negative_after_CPU"] <= -0.1
    for k, v in m.gate.state_dict().items():
        if k != "2.bias":
            assert torch.equal(v, before[k])


def test_certificate_allows_open_gate_when_outputs_identical_and_counts_actual_forwards():
    m = model()
    # First row is meaningful negative; second protected row is an identical R2 output.
    # Override one parent gate to distinguish rows using the first feature.
    with torch.no_grad():
        g = m.parent.parent.gate
        g[0].weight.zero_()
        g[0].bias.zero_()
        g[0].weight[0, 0] = 1
        g[-1].weight.zero_()
        g[-1].weight[0, 0] = -1
        g[-1].bias.zero_()
        m.parent.parent.net[-1].bias.fill_(0.5)
        m.gate[0].weight.zero_()
        m.gate[0].bias.zero_()
        m.gate[0].weight[0, 0] = 1
        m.gate[-1].weight.zero_()
        m.gate[-1].weight[0, 0] = 1
        m.gate[-1].bias.fill_(-0.2)
    x = np.zeros((2, 288), np.float32)
    x[1, 0] = 1
    with torch.no_grad():
        prior = torch.cat([m.parent(torch.from_numpy(row).reshape(1, 288)) for row in x]).numpy()
    arrays = dict(
        features=x,
        parent_delta=prior,
        teacher=np.zeros(2, bool),
        protected_nonzero=np.array([True, False]),
        positive_core=np.zeros(2, bool),
    )
    cert = router_certificate(m, arrays)
    assert cert["protected_gate_open_rows"] == 1
    assert cert["all_protected_predictions_exact_parent"]
    assert cert["CPU_parent_forward_rows"] == 4 and cert["CPU_router_rows"] == 2


@pytest.mark.parametrize("seed", [True, -1, 3, None, "0"])
def test_only_preregistered_integer_seeds(seed):
    with pytest.raises(ValueError):
        router_config(seed)
