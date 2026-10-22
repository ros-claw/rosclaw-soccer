from types import SimpleNamespace

import numpy as np
import pytest
import torch

from rosclaw_soccer.rsi.paid_run_teacher_audit import replay_teacher_world


@pytest.mark.parametrize("fault", ["legacy", "no_adapter", "standing", "complete_reference"])
def test_interrupted_prefix_is_not_a_general_healthy_suffix_bypass(fault):
    actor = torch.nn.Linear(1, 1).eval().requires_grad_(False)
    ep = SimpleNamespace(
        env=SimpleNamespace(model=SimpleNamespace(opt=SimpleNamespace(timestep=0.002)), data=None),
        clip=SimpleNamespace(length=3),
    )
    kick = dict(qpos=np.zeros((2, 10, 43)))
    stand = None

    def adapter(actual, nominal, evidence):
        return nominal, 0

    width = 256
    if fault == "legacy":
        width = 128
    if fault == "no_adapter":
        adapter = None
    if fault == "standing":
        stand = dict(qpos=np.zeros((1, 10, 43)))
    if fault == "complete_reference":
        ep.clip.length = 2
    counts = dict(native=0, neural=0, kinematic_intent=0)
    with pytest.raises(ValueError):
        replay_teacher_world(
            ep,
            actor,
            kick,
            stand,
            latent=np.zeros((2, 6)),
            center=105,
            foot="left",
            counts=counts,
            hidden_width=width,
            action_adapter=adapter,
            allow_incomplete_healthy_prefix=True,
        )
    assert counts == dict(native=0, neural=0, kinematic_intent=0)
