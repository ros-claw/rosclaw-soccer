import torch
from test_dual_recurrent_motor import ToyBase
from test_paid_ancestor_router_student import model

from rosclaw_soccer.rsi.paid_ancestor_router_student import FrozenAncestralRouterActor
from scripts.rsi_run_shared_router_relay import freeze_router


def test_freeze_disables_all_gradients_without_changing_actor():
    actor = FrozenAncestralRouterActor(ToyBase(), model(), torch.full((29,), 0.5)).eval()
    assert any(p.requires_grad for p in actor.parameters())
    before = {k: v.clone() for k, v in actor.state_dict().items()}
    assert freeze_router(actor) is actor
    assert not actor.training and not any(p.requires_grad for p in actor.parameters())
    assert all(torch.equal(v, before[k]) for k, v in actor.state_dict().items())
