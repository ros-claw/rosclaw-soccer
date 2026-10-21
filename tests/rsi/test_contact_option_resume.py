"""Synthetic checkpoint contracts; CUDA RNG is mocked, no physical evidence."""

import copy
import hashlib

import pytest
import torch
from torch import nn

from rosclaw_soccer.rsi.contact_option_resume import IDENTITY_KEYS, restore_contact_option_boundary
from rosclaw_soccer.rsi.native_contact_adapter import NativeResidualContactActor
from rosclaw_soccer.rsi.native_contact_ppo import make_contact_actor_critic


class Base(nn.Module):
    reference_features = 79

    def __init__(self):
        super().__init__()
        self.head = nn.Linear(177, 29)
        self.register_buffer("prior_mean", torch.zeros(1, 98))
        self.register_buffer("prior_denominator", torch.ones(1, 98))

    def forward(self, x):
        return self.head(x)


def adam(policy):
    return torch.optim.Adam(
        [
            dict(params=[p for p in policy.actor.parameters() if p.requires_grad], lr=3e-4),
            dict(params=list(policy.critic.parameters()), lr=3e-4),
            dict(params=[policy.log_std], lr=1e-5),
        ]
    )


def checkpoint():
    torch.manual_seed(7607)
    policy = make_contact_actor_critic(NativeResidualContactActor(Base()))
    fresh = copy.deepcopy(policy)
    optimizer = adam(policy)
    for group in optimizer.param_groups:
        for p in group["params"]:
            p.grad = torch.ones_like(p)
    optimizer.step()
    opt = optimizer.state_dict()
    for gi, g in enumerate(opt["param_groups"]):
        for i in g["params"]:
            opt["state"][i]["step"].fill_(320 if gi == 1 else 256)
    state = dict(
        schema="soccer.rsi.native_contact24_recovery60_residual_ppo.v1",
        protocol="native-contact24-recovery60-v1",
        training_seed=7607,
        episode_count=20,
        course_episode=19,
        fresh_optimizer=True,
        activation_ceiling="SIM_ONLY",
        hardware_authorized=False,
        promotion_authorized=False,
        counts=dict(
            optimizer_attempted=320, optimizer_completed=320, value_only_steps=64, ppo_steps=256
        ),
        policy_state_dict=policy.state_dict(),
        optimizer_state_dict=opt,
        torch_rng=torch.get_rng_state(),
        cuda_rng=[torch.get_rng_state().clone()],
    )
    identity = {k: "sha256:" + "a" * 64 for k in IDENTITY_KEYS}
    state.update(identity)
    return fresh, state, identity


def test_restore_preserves_all_weights_moments_and_rng(monkeypatch, tmp_path):
    policy, state, identity = checkpoint()
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    seen = []
    monkeypatch.setattr(torch.cuda, "set_rng_state_all", lambda x: seen.extend(x))
    path = tmp_path / "state.pt"
    torch.save(state, path)
    optimizer = adam(policy)
    result = restore_contact_option_boundary(
        policy,
        optimizer,
        path,
        expected_hash="sha256:" + hashlib.sha256(path.read_bytes()).hexdigest(),
        expected_identity=identity,
    )
    assert result["course_episode"] == 19 and result["optimizer_resumed"]
    assert all(
        torch.equal(v, state["policy_state_dict"][k]) for k, v in policy.state_dict().items()
    )
    assert torch.equal(torch.get_rng_state(), state["torch_rng"])
    assert torch.equal(seen[0], state["cuda_rng"][0])
    assert len(optimizer.state) == 13


@pytest.mark.parametrize("defect", ["empty_moments", "bad_step", "count", "identity", "foundation"])
def test_refuses_incomplete_or_rebound_resume(monkeypatch, tmp_path, defect):
    policy, state, identity = checkpoint()
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    if defect == "empty_moments":
        state["optimizer_state_dict"]["state"] = {}
    if defect == "bad_step":
        next(iter(state["optimizer_state_dict"]["state"].values()))["step"].fill_(1)
    if defect == "count":
        state["counts"]["optimizer_completed"] = 319
    if defect == "identity":
        identity["native_model_hash"] = "wrong"
    if defect == "foundation":
        state["policy_state_dict"]["actor.base.head.bias"] += 1
    path = tmp_path / "bad.pt"
    torch.save(state, path)
    with pytest.raises(ValueError):
        restore_contact_option_boundary(
            policy,
            adam(policy),
            path,
            expected_hash="sha256:" + hashlib.sha256(path.read_bytes()).hexdigest(),
            expected_identity=identity,
        )
