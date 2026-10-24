from copy import deepcopy

import pytest
import torch
from torch import nn

from rosclaw_soccer.rsi.refinement_motor_learning import (
    COMMITMENTS,
    SCHEMA,
    refinement_config,
    refinement_schedule,
    validate_refinement_checkpoint,
)


def records():
    return [
        dict(
            index=i,
            role="retained_original17"
            if i < 17
            else "retained_current41"
            if i < 58
            else "qualified_rescue",
        )
        for i in range(62)
    ]


def fixture():
    head = nn.Sequential(nn.Linear(288, 128), nn.Tanh(), nn.Linear(128, 29))
    expected = dict(head.state_dict())
    initial = {k: v.clone() for k, v in expected.items()}
    initial["2.weight"].zero_()
    initial["2.bias"].zero_()
    commitments = dict.fromkeys(COMMITMENTS, "sha256:" + "1" * 64)
    cfg = refinement_config(115023, "cpu")
    state = dict(
        schema=SCHEMA,
        config=cfg,
        initial_head=initial,
        head={k: v + 0.01 for k, v in initial.items()},
        new_head_steps=2048,
        head_optimizer=dict(
            param_groups=[
                dict(
                    params=list(range(4)),
                    lr=cfg["lr"],
                    weight_decay=0,
                    amsgrad=False,
                    maximize=False,
                )
            ],
            state={
                i: dict(
                    step=torch.tensor(2048.0),
                    exp_avg=torch.zeros_like(v),
                    exp_avg_sq=torch.zeros_like(v),
                )
                for i, v in enumerate(expected.values())
            },
        ),
        **commitments,
    )
    return state, expected, cfg, commitments


def test_schedule_covers_all_disjoint_whole_histories_and_seeds_differ():
    r = records()
    a = refinement_schedule(r, 115023)
    assert len(a) == 2048 and all(len(v) == 10 for v in a)
    assert set(sum(a, [])) == set(range(62))
    assert a == refinement_schedule(r, 115023)
    assert a != refinement_schedule(r, 115024)
    assert all(
        sum(i < 17 for i in b) == 4
        and sum(17 <= i < 58 for i in b) == 4
        and sum(i >= 58 for i in b) == 2
        for b in a
    )


def test_valid_synthetic_checkpoint_contract_not_training_evidence():
    s, e, c, h = fixture()
    validate_refinement_checkpoint(s, e, config=c, commitments=h)


@pytest.mark.parametrize(
    "defect",
    ["steps", "ancestry", "shape", "nan", "initial", "unchanged", "optimizer", "old4096", "extra"],
)
def test_checkpoint_violations_rejected(defect):
    s, e, c, h = fixture()
    if defect == "steps":
        s["new_head_steps"] = 0
    elif defect == "ancestry":
        s["parent_training_hash"] = "changed"
    elif defect == "shape":
        s["head"]["0.weight"] = torch.zeros(1)
    elif defect == "nan":
        s["head"]["0.weight"][0, 0] = float("nan")
    elif defect == "initial":
        s["initial_head"]["2.bias"][0] = 0.1
    elif defect == "unchanged":
        s["head"]["2.bias"] = s["initial_head"]["2.bias"].clone()
    elif defect == "optimizer":
        s["head_optimizer"]["param_groups"][0]["params"].append(4)
    elif defect == "old4096":
        s["head_optimizer"]["state"][0]["step"] = torch.tensor(4096.0)
    else:
        s["arbitrary_pickle_object"] = "forbidden"
    with pytest.raises(ValueError):
        validate_refinement_checkpoint(s, e, config=c, commitments=h)


def test_record_role_corruption_rejected():
    r = deepcopy(records())
    r[0]["role"] = "qualified_rescue"
    with pytest.raises(ValueError):
        refinement_schedule(r, 115023)
