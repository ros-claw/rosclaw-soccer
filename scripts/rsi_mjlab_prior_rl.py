"""Optional RSL-RL adapters for SIM prior initialization and explicit rehearsal.

Construct the actor before PPO allocates its optimizer. Rehearsal is one
ordinary, separately counted supervised optimizer step after each PPO update;
it is not a substituted gradient, an evaluation pass, or a success guarantee.
"""

from pathlib import Path

import numpy as np
import torch
from rsl_rl.algorithms import PPO
from rsl_rl.models import MLPModel
from tensordict import TensorDict

from rosclaw_soccer.providers.g1.velocity_policy import file_hash
from rosclaw_soccer.rsi.velocity_prior_actor import VelocityPriorActor


class VelocityPriorMLPModel(MLPModel):
    def __init__(
        self,
        *args,
        prior_model,
        prior_model_hash,
        prior_checkpoint,
        prior_checkpoint_hash,
        **kwargs,
    ):
        # MJLab strips disabled CNN config before RSL construction. Direct
        # evaluation reads the committed pre-normalization configuration.
        if kwargs.pop("cnn_cfg", None) is not None:
            raise ValueError("velocity prior is an MLP, not a CNN model")
        if kwargs.get("obs_normalization", False):
            raise ValueError("fixed qualified prior normalization must not be replaced by RMS")
        checkpoint = Path(prior_checkpoint)
        if file_hash(checkpoint) != prior_checkpoint_hash:
            raise ValueError("immutable qualified initialization checkpoint required")
        super().__init__(*args, **kwargs)
        if self.obs_dim != 168 or self.distribution is None or self.distribution.input_dim != 29:
            raise ValueError("168D prior actor and scalar-std Gaussian29 required")
        prior = VelocityPriorActor.from_onnx(Path(prior_model), expected_hash=prior_model_hash)
        loaded = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if (
            loaded["optimizer_steps"] != 0
            or loaded["velocity_features"] != 98
            or loaded["reference_features"] != 70
            or loaded["original_onnx_hash"] != prior_model_hash
        ):
            raise ValueError("qualified unchanged prior initialization schema required")
        prior.load_state_dict(loaded["actor_state_dict"], strict=True)
        # This happens in actor construction, before PPO constructs Adam.
        self.mlp = prior
        self.obs_normalizer = torch.nn.Identity()
        self.obs_normalization = False


def load_rehearsal(path, expected_hash, device, reference_features=70):
    if type(reference_features) is not int or reference_features not in (70, 79):
        raise ValueError("explicit reference70 or reference70+perception9 rehearsal ABI required")
    path = Path(path)
    if file_hash(path) != expected_hash:
        raise ValueError("immutable recorded native rehearsal bank required")
    with np.load(path, allow_pickle=False) as raw:
        if set(raw.files) != {"observation", "action"}:
            raise ValueError("explicit native observation/action rehearsal schema required")
        obs = raw["observation"].copy()
        actions = raw["action"].copy()
    if (
        obs.shape != (1200, 98)
        or actions.shape != (1200, 29)
        or obs.dtype != np.float32
        or actions.dtype != np.float32
        or not np.isfinite(obs).all()
        or not np.isfinite(actions).all()
    ):
        raise ValueError("complete finite actual native 1200-row rehearsal bank required")
    padded = np.concatenate((obs, np.zeros((1200, reference_features), dtype=np.float32)), axis=1)
    return TensorDict(
        {"actor": torch.from_numpy(padded).to(device)}, batch_size=[1200]
    ), torch.from_numpy(actions).to(device)


class RehearsalPPO(PPO):
    def __init__(
        self,
        *args,
        rehearsal_path,
        rehearsal_hash,
        rehearsal_weight=1.0,
        rehearsal_batch_size=256,
        rehearsal_reference_features=70,
        **kwargs,
    ):
        if (
            type(rehearsal_weight) is not float
            or not 0 < rehearsal_weight <= 10
            or type(rehearsal_batch_size) is not int
            or not 1 <= rehearsal_batch_size <= 1200
        ):
            raise ValueError("bounded explicit rehearsal weight and batch required")
        super().__init__(*args, **kwargs)
        if self.is_multi_gpu:
            raise ValueError("initial rehearsal protocol is single-GPU only")
        self.rehearsal_obs, self.rehearsal_actions = load_rehearsal(
            rehearsal_path, rehearsal_hash, self.device, rehearsal_reference_features
        )
        self.rehearsal_reference_features = rehearsal_reference_features
        self.rehearsal_hash = rehearsal_hash
        self.rehearsal_weight = rehearsal_weight
        self.rehearsal_batch_size = rehearsal_batch_size
        self.rehearsal_updates = 0
        self.rehearsal_attempted_updates = 0
        self.rehearsal_rows = 0

    def update(self):
        losses = super().update()
        # Deterministic cyclic sampling; each old physical state remains a
        # rehearsal example, not a claimed independent new rollout.
        indices = (
            torch.arange(self.rehearsal_batch_size, device=self.device)
            + self.rehearsal_updates * self.rehearsal_batch_size
        ) % 1200
        prediction = self.actor(self.rehearsal_obs[indices], stochastic_output=False)
        loss = torch.nn.functional.mse_loss(prediction, self.rehearsal_actions[indices])
        if not torch.isfinite(loss):
            raise ValueError("finite explicit old-skill rehearsal objective required")
        self.optimizer.zero_grad(set_to_none=True)
        (self.rehearsal_weight * loss).backward()
        norm = torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        if not torch.isfinite(norm):
            raise ValueError("finite clipped rehearsal gradients required")
        self.rehearsal_attempted_updates += 1
        self.optimizer.step()
        self.rehearsal_updates += 1
        self.rehearsal_rows += self.rehearsal_batch_size
        losses["native_rehearsal_mse"] = float(loss.detach())
        return losses

    def save(self):
        state = super().save()
        state["native_rehearsal"] = dict(
            bank_hash=self.rehearsal_hash,
            weight=self.rehearsal_weight,
            batch_size=self.rehearsal_batch_size,
            updates=self.rehearsal_updates,
            attempted_updates=self.rehearsal_attempted_updates,
            rows=self.rehearsal_rows,
        )
        if self.rehearsal_reference_features != 70:
            state["native_rehearsal"]["reference_features"] = self.rehearsal_reference_features
        return state

    def load(self, loaded_dict, load_cfg=None, strict=True):
        state = loaded_dict["native_rehearsal"]
        if (
            state["bank_hash"] != self.rehearsal_hash
            or state.get("reference_features", 70) != self.rehearsal_reference_features
            or state["weight"] != self.rehearsal_weight
            or state["batch_size"] != self.rehearsal_batch_size
            or state["updates"] != state["attempted_updates"]
            or state["rows"] != state["updates"] * self.rehearsal_batch_size
        ):
            raise ValueError("same committed rehearsal protocol and completed updates required")
        result = super().load(loaded_dict, load_cfg, strict)
        self.rehearsal_updates = state["updates"]
        self.rehearsal_attempted_updates = state["attempted_updates"]
        self.rehearsal_rows = state["rows"]
        return result
