"""Optional 177D actor/304D critic construction BEFORE a new PPO optimizer."""

from pathlib import Path

import torch
from rsl_rl.models import MLPModel

from rosclaw_soccer.providers.g1.velocity_policy import file_hash
from rosclaw_soccer.rsi.ball_aware_migration import expand_ball_aware_state
from rosclaw_soccer.rsi.velocity_prior_actor import VelocityPriorActor


def load_parent(path, expected):
    path = Path(path)
    if file_hash(path) != expected:
        raise ValueError("immutable parent checkpoint required")
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state.get("actor_observation_dim") != 168 or state.get("stable_prior_rehearsal") is not True:
        raise ValueError("qualified original 168D SIM parent schema required")
    return state


class BallAwarePriorMLPModel(MLPModel):
    def __init__(
        self,
        *args,
        prior_model,
        prior_model_hash,
        parent_checkpoint,
        parent_checkpoint_hash,
        **kwargs,
    ):
        if kwargs.pop("cnn_cfg", None) is not None or kwargs.get("obs_normalization", False):
            raise ValueError("original fixed-normalization MLP required")
        super().__init__(*args, **kwargs)
        if self.obs_dim != 177 or self.distribution is None or self.distribution.input_dim != 29:
            raise ValueError("new ball-aware177 actor with Gaussian29 required")
        parent = load_parent(parent_checkpoint, parent_checkpoint_hash)
        actor = VelocityPriorActor.from_onnx(
            Path(prior_model), expected_hash=prior_model_hash, reference_features=79
        )
        state = expand_ball_aware_state(parent["algorithm"]["actor_state_dict"], part="actor")
        actor.load_state_dict({k: state["mlp." + k] for k in actor.state_dict()}, strict=True)
        self.mlp = actor
        self.obs_normalizer = torch.nn.Identity()
        self.obs_normalization = False
        self.load_state_dict(state, strict=True)


class BallAwareCriticMLPModel(MLPModel):
    def __init__(self, *args, parent_checkpoint, parent_checkpoint_hash, **kwargs):
        if kwargs.pop("cnn_cfg", None) is not None or kwargs.get("obs_normalization") is not True:
            raise ValueError("original critic normalization plus new nine channels required")
        super().__init__(*args, **kwargs)
        if self.obs_dim != 304:
            raise ValueError("new explicit ball-aware304 critic required")
        parent = load_parent(parent_checkpoint, parent_checkpoint_hash)
        self.load_state_dict(
            expand_ball_aware_state(parent["algorithm"]["critic_state_dict"], part="critic"),
            strict=True,
        )
