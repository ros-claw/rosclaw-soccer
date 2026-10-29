import numpy as np
import pytest
import torch
from test_dual_recurrent_motor import ToyBase
from test_paid_ancestor_router_student import model as router_model
from test_shared_native_neural_player import players

from rosclaw_soccer.rsi.paid_ancestor_router_student import FrozenAncestralRouterActor
from rosclaw_soccer.rsi.shared_native_neural_player import SharedNativeNeuralPlayer


def build(w, existing, actor, prefix, **kwargs):
    p = existing[prefix]
    return SharedNativeNeuralPlayer(
        w,
        prefix,
        actor,
        clip=p.clip,
        torso_index=p.torso_index,
        build_observation=p.build_observation,
        goal=p.goal,
        offset=p.offset,
        scale=p.scale,
        order=p.order,
        **kwargs,
    )


def test_two_private_router_histories_and_real_shared_execution():
    w, existing = players()
    actor = FrozenAncestralRouterActor(ToyBase(), router_model(), torch.full((29,), 0.5))
    actor.requires_grad_(False).eval()
    a = build(w, existing, actor, "a/")
    b = build(w, existing, actor, "b/")
    assert a.state[0].shape == (2, 1, 128) and a.state[0].data_ptr() != b.state[0].data_ptr()
    ta, trace_a = a.propose()
    tb, _ = b.propose()
    with torch.no_grad():
        direct, after = actor(torch.from_numpy(trace_a["NN_obs"]).reshape(1, 1, 160), a.state)
    np.testing.assert_array_equal(trace_a["NN_action"], direct[0, 0].numpy())
    np.testing.assert_array_equal(trace_a["NN_h_after"], after[0].numpy())
    w.advance({"a/": ta, "b/": tb})
    a.commit()
    b.commit()
    assert a.decisions == b.decisions == 1
    np.testing.assert_array_equal(a.last_action, trace_a["NN_action"])


def test_router_refuses_wrong_codec_live_history_and_teacher_authority():
    w, existing = players()
    wrong = FrozenAncestralRouterActor(ToyBase(), router_model(), torch.ones(29))
    wrong.requires_grad_(False).eval()
    with pytest.raises(ValueError, match="action scale"):
        build(w, existing, wrong, "a/")
    actor = FrozenAncestralRouterActor(ToyBase(), router_model(), torch.full((29,), 0.5))
    actor.requires_grad_(False).eval()
    first = build(w, existing, actor, "a/")
    with pytest.raises(ValueError, match="teacher authority"):
        first.propose(with_reference_evidence=True)
    w.advance({p: existing[p].propose()[0] for p in w.layouts})
    with pytest.raises(ValueError, match="own 128-wide history"):
        build(w, existing, actor, "a/")
