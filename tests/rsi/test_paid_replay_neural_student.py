import copy

import pytest
import torch

from rosclaw_soccer.rsi.paid_replay_neural import FIT_STATUS, ReplayMotorHead, replay_config
from rosclaw_soccer.rsi.paid_replay_neural_student import validate_neural_state


def fixture():
    cfg = replay_config(0)
    torch.manual_seed(cfg["seed"])
    head = ReplayMotorHead(torch.zeros(288), torch.ones(288))
    template = copy.deepcopy(head.state_dict())
    model = copy.deepcopy(template)
    params = {k: v for k, v in template.items() if k.startswith("net.")}
    for k in params:
        model[k] += 0.001
    counts = dict(
        completed_updates=6144,
        nonzero_gradient_updates=6144,
        active_forward_rows=6144 * 512,
        rehearsal_forward_rows=6144 * 512,
        changed_network_tensors=list(params),
    )
    state = dict(
        schema="paid.soft_replay_neural.offline.v1",
        config=cfg,
        source_head="frozen",
        data_hash="data",
        body_scene_hash="body",
        parent_hash="parent",
        parent_checkpoint_hash="cp",
        declaration_hash="decl",
        counts=counts,
        certificate={},
        model=model,
        initial_model=copy.deepcopy(template),
        optimizer=dict(
            param_groups=[dict(params=list(range(8)), lr=cfg["final_lr"], weight_decay=0)],
            state={
                i: dict(
                    step=torch.tensor(6144.0),
                    exp_avg=torch.zeros_like(v),
                    exp_avg_sq=torch.ones_like(v),
                )
                for i, v in enumerate(params.values())
            },
        ),
    )
    receipt = {
        k: copy.deepcopy(state[k])
        for k in (
            "config",
            "source_head",
            "data_hash",
            "body_scene_hash",
            "parent_hash",
            "parent_checkpoint_hash",
            "declaration_hash",
            "counts",
            "certificate",
        )
    }
    receipt.update(
        status=FIT_STATUS,
        optimizer_updates=6144,
        own_native_steps=0,
        runtime_admitted=False,
        weights_only_checkpoint_reload=True,
        parent_parameters_exact_unchanged=True,
        hardware_authorized=False,
        promotion_authorized=False,
    )
    return state, receipt, template


def test_exact_offline_fit_and_optimizer_contract():
    validate_neural_state(*fixture())


@pytest.mark.parametrize(
    "bad", ["count", "normalizer", "nan", "initial", "moment", "step", "runtime", "keys"]
)
def test_rejects_forged_counts_state_or_admission(bad):
    s, r, t = fixture()
    if bad == "count":
        s["counts"]["completed_updates"] = r["counts"]["completed_updates"] = 1
    elif bad == "normalizer":
        s["model"]["feature_scale"][0] = 2
    elif bad == "nan":
        s["model"]["net.0.weight"][0, 0] = float("nan")
    elif bad == "initial":
        s["initial_model"]["net.0.bias"][0] += 1
    elif bad == "moment":
        s["optimizer"]["state"][0]["exp_avg_sq"][0, 0] = -1
    elif bad == "step":
        s["optimizer"]["state"][0]["step"] = torch.tensor(0.0)
    elif bad == "runtime":
        r["runtime_admitted"] = True
    else:
        s["model"]["extra"] = torch.zeros(1)
    with pytest.raises(ValueError):
        validate_neural_state(s, r, t)
