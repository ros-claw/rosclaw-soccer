import pytest
import torch

from rosclaw_soccer.rsi.paid_contact_residual import ContactResidual
from rosclaw_soccer.rsi.paid_gate_retention import project_contact_gate


def fixture():
    model = ContactResidual()
    with torch.no_grad():
        model.gate[-1].bias.fill_(-1)
    old = {k: p.detach().clone() for k, p in model.gate.named_parameters()}
    return model, old, torch.zeros(9, 288)


def test_feasible_update_remains_exact_proposal():
    m, old, x = fixture()
    with torch.no_grad():
        m.gate[-1].bias.fill_(-0.5)
    proposed = {k: p.clone() for k, p in m.gate.named_parameters()}
    r = project_contact_gate(m.gate, old, x)
    assert r["accepted_fraction"] == 1
    assert r["gate_forward_rows"] == 9
    assert all(torch.equal(p, proposed[k]) for k, p in m.gate.named_parameters())


def test_mean_loss_cannot_hide_one_retained_state_gate_violation():
    m, old, x = fixture()
    # Directly route one retained feature through the gate; the other eight
    # states remain negative. Average score would hide this one intervention.
    with torch.no_grad():
        m.gate[0].weight.zero_()
        m.gate[0].bias.zero_()
        m.gate[0].weight[0, 0] = 1
        m.gate[-1].weight.zero_()
        m.gate[-1].bias.fill_(-1)
    old = {k: p.detach().clone() for k, p in m.gate.named_parameters()}
    x[-1, 0] = 1
    delta_head = {k: p.clone() for k, p in m.net.named_parameters()}
    with torch.no_grad():
        m.gate[-1].weight[0, 0] = 2
    assert m.gate(x).mean() < -0.5 and m.gate(x).max() > 0
    r = project_contact_gate(m.gate, old, x)
    assert 0 < r["accepted_fraction"] < 1
    assert m.gate(x).max() <= -0.1
    assert all(torch.equal(p, delta_head[k]) for k, p in m.net.named_parameters())
    assert not r["optimizer_moments_rewound"]


def test_infeasible_proposal_can_restore_exact_previous_gate():
    m, old, x = fixture()
    with torch.no_grad():
        m.gate[-1].bias.fill_(1)
    r = project_contact_gate(m.gate, old, x, max_halvings=0)
    assert r["gate_update_rejected"] and r["accepted_fraction"] == 0
    assert all(torch.equal(p, old[k]) for k, p in m.gate.named_parameters())


@pytest.mark.parametrize(
    "fault",
    ["nan", "dtype", "shape", "empty", "old_nan", "keys", "margin", "unbounded", "old_infeasible"],
)
def test_corrupt_or_inadmissible_retention_input_rejected(fault):
    m, old, x = fixture()
    if fault == "nan":
        x[0, 0] = float("nan")
    elif fault == "dtype":
        x = x.double()
    elif fault == "shape":
        x = torch.zeros(9, 160)
    elif fault == "empty":
        x = torch.zeros(0, 288)
    elif fault == "old_nan":
        old["2.bias"][0] = float("nan")
    elif fault == "keys":
        old.pop("2.bias")
    elif fault == "old_infeasible":
        old["2.bias"].fill_(1)
        with torch.no_grad():
            m.gate[-1].bias.fill_(1)
    with pytest.raises(ValueError):
        project_contact_gate(
            m.gate,
            old,
            x,
            margin=float("nan") if fault == "margin" else 0.1,
            max_halvings=9 if fault == "unbounded" else 8,
        )
