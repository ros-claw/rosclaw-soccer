"""Conservative per-frame latent Gaussian PPO-Clip prototype, SIM data only.

Monte Carlo terminal returns (gamma=lambda=1) train the actor and state critic.
The latent Gaussian precedes tanh, deterministic slew and joint shielding;
shielded output is NOT assigned a Gaussian likelihood. Old successful states
regularize mean-policy drift; this is not a mathematical retention guarantee.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.rsi import step_motor_network as warm_network
from rosclaw_soccer.rsi.stochastic_step_execution import raw_actor_mean
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json

SCHEMA = "soccer.rsi.online_step_motor_mc_ppo.v1"
AUTHORITY_FLAGS = (
    "physics_qualified",
    "online_rl_qualified",
    "promotion_authorized",
    "runtime_execution_authorized",
    "hardware_authorized",
    "fresh_holdout_open_authorized",
)


def validate_model(model: dict[str, Any]) -> None:
    if (
        model.get("schema") != SCHEMA
        or model.get("activation_ceiling") != "SIM_ONLY"
        or model.get("learning_kind") != "per_frame_latent_gaussian_mc_ppo_clip_prototype"
        or model.get("source_hash") != hash_bytes(Path(__file__).read_bytes())
        or any(model.get(k) is not False for k in AUTHORITY_FLAGS)
        or model.get("model_hash")
        != hash_json({k: v for k, v in model.items() if k != "model_hash"})
    ):
        raise ValueError("unsealed unqualified online motor prototype")
    warm_network.validate_model(model["warm_start_model"])
    # Validate numeric architecture without rewriting provenance of the actual model.
    numeric = numeric_view(model)
    warm_network.validate_model(numeric)
    receipt = model["learning_receipt"]
    if (
        receipt.get("algorithm") != "PPO_CLIP_MC_TERMINAL"
        or receipt.get("likelihood") != "unshielded Gaussian before tanh"
        or receipt.get("gamma") != 1.0
        or receipt.get("gae_lambda") != 1.0
        or receipt.get("physical_batch_hash") != model["physical_batch_hash"]
        or not isinstance(model["physical_batch_hash"], str)
        or not model["physical_batch_hash"].startswith("sha256:")
    ):
        raise ValueError("explicit physical policy-gradient receipt required")


def numeric_view(model: dict[str, Any]) -> dict[str, Any]:
    """Internal old-architecture decoder ONLY; never an exported training artifact."""
    view = dict(model["warm_start_model"])
    view["actor"] = model["actor"]
    view["critic"] = model["critic"]
    view["numeric_decoder_view_only"] = True
    view.pop("model_hash")
    view["model_hash"] = hash_json(view)
    return view


def fit_update(
    warm: dict[str, Any],
    arrays: Any,
    *,
    batch_hash: str,
    teacher_features: Any,
    epochs: int = 4,
    learning_rate: float = 3e-5,
) -> dict[str, Any]:
    import torch

    warm_network.validate_model(warm)
    x = np.asarray(arrays["observation"], dtype=np.float64)
    z = np.asarray(arrays["latent_action"], dtype=np.float64)
    old_logp = np.asarray(arrays["old_log_probability"], dtype=np.float64)
    returns = np.asarray(arrays["terminal_return"], dtype=np.float64)
    std = np.asarray(arrays["std_raw"], dtype=np.float64)
    groups = np.asarray(arrays["trajectory_index"])
    teachers = np.asarray(teacher_features, dtype=np.float64)
    n = len(x)
    if (
        n < 100
        or x.shape != (n, 134)
        or z.shape != (n, 12)
        or any(v.shape != (n,) for v in (old_logp, returns, std, groups))
        or teachers.ndim != 2
        or teachers.shape[1] != 134
        or len(teachers) < 100
        or not all(np.isfinite(v).all() for v in (x, z, old_logp, returns, std, groups, teachers))
        or np.any((std < 0.01) | (std > 0.15))
        or not isinstance(batch_hash, str)
        or not batch_hash.startswith("sha256:")
        or type(epochs) is not int
        or not 1 <= epochs <= 10
        or not 1e-6 <= learning_rate <= 1e-4
    ):
        raise ValueError("aligned finite actual physical rollout batch required")
    old_means = np.clip((x - np.asarray(warm["mean"])) / np.asarray(warm["scale"]), -8, 8)
    for index, layer in enumerate(warm["actor"]["layers"]):
        old_means = old_means @ np.asarray(layer["weight"]).T + np.asarray(layer["bias"])
        if index < len(warm["actor"]["layers"]) - 1:
            old_means = np.tanh(old_means)
    if not np.allclose(old_means[0], raw_actor_mean(warm, x[0]), atol=1e-12, rtol=0):
        raise ValueError("batched mean differs from causal inference decoder")
    reconstructed = np.sum(
        -0.5 * ((z - old_means) / std[:, None]) ** 2
        - np.log(std[:, None])
        - 0.5 * np.log(2 * np.pi),
        axis=1,
    )
    if not np.allclose(old_logp, reconstructed, atol=1e-8, rtol=0):
        raise ValueError("actual rollout latent likelihood differs from old actor")
    if any(
        not np.all(returns[groups == g] == returns[groups == g][0]) for g in set(groups.tolist())
    ):
        raise ValueError("one measured Monte Carlo terminal return per trajectory required")
    torch.set_num_threads(4)
    torch.manual_seed(20261001317)
    torch.use_deterministic_algorithms(True)

    def load_net(name: str) -> Any:
        modules: list[Any] = []
        layers = warm[name]["layers"]
        for i, layer in enumerate(layers):
            weight = torch.tensor(layer["weight"], dtype=torch.float32)
            linear = torch.nn.Linear(weight.shape[1], weight.shape[0])
            with torch.no_grad():
                linear.weight.copy_(weight)
                linear.bias.copy_(torch.tensor(layer["bias"], dtype=torch.float32))
            modules.append(linear)
            if i < len(layers) - 1:
                modules.append(torch.nn.Tanh())
        return torch.nn.Sequential(*modules)

    actor, critic = load_net("actor"), load_net("critic")
    frozen_actor = copy.deepcopy(actor).requires_grad_(False)
    mean, scale = np.asarray(warm["mean"]), np.asarray(warm["scale"])
    obs = torch.tensor(np.clip((x - mean) / scale, -8, 8), dtype=torch.float32)
    teacher_obs = torch.tensor(np.clip((teachers - mean) / scale, -8, 8), dtype=torch.float32)
    action = torch.tensor(z, dtype=torch.float32)
    noise_scale = torch.tensor(std[:, None], dtype=torch.float32)
    old_mean = frozen_actor(obs).detach()
    # Float32 execution has different last bits from the float64 audited density.
    # Use old means at the optimizer's precision so initial ratio is exactly one.
    old_density = (
        -0.5 * ((action - old_mean) / noise_scale).square()
        - noise_scale.log()
        - 0.5 * np.log(2 * np.pi)
    ).sum(dim=1)
    value_targets = torch.tensor(
        (returns - warm["critic_return_mean"]) / warm["critic_return_scale"],
        dtype=torch.float32,
    )
    with torch.no_grad():
        advantage = value_targets - critic(obs)[:, 0]
        advantage = (advantage - advantage.mean()) / advantage.std(unbiased=False).clamp_min(1e-6)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=learning_rate)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=learning_rate)
    rng = np.random.default_rng(20261001317)
    steps = 0
    stopped = False
    final_kl = 0.0
    for _ in range(epochs):
        for offset_ids in np.array_split(rng.permutation(n), max(1, (n + 1023) // 1024)):
            ids = torch.tensor(offset_ids, dtype=torch.int64)
            logp = (
                -0.5 * ((action[ids] - actor(obs[ids])) / noise_scale[ids]).square()
                - noise_scale[ids].log()
                - 0.5 * np.log(2 * np.pi)
            ).sum(dim=1)
            ratio = torch.exp(logp - old_density[ids])
            surrogate = torch.minimum(
                ratio * advantage[ids],
                torch.clamp(ratio, 0.8, 1.2) * advantage[ids],
            ).mean()
            teacher_ids = rng.choice(len(teachers), size=min(1024, len(teachers)), replace=False)
            replay = teacher_obs[teacher_ids]
            replay_loss = (actor(replay) - frozen_actor(replay)).square().mean()
            actor_loss = -surrogate + 2.0 * replay_loss
            actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
            # Exact equal-variance Gaussian KL on every actual rollout state.
            previous_state = copy.deepcopy(actor.state_dict())
            actor_optimizer.step()
            with torch.no_grad():
                final_kl = float(
                    ((actor(obs) - old_mean).square() / (2 * noise_scale.square()))
                    .sum(dim=1)
                    .mean()
                )
            if final_kl > 0.02:
                actor.load_state_dict(previous_state)
                with torch.no_grad():
                    final_kl = float(
                        ((actor(obs) - old_mean).square() / (2 * noise_scale.square()))
                        .sum(dim=1)
                        .mean()
                    )
                stopped = True
                break
            critic_optimizer.zero_grad()
            critic_loss = (critic(obs[ids])[:, 0] - value_targets[ids]).square().mean()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
            critic_optimizer.step()
            steps += 1
        if stopped:
            break

    def export(net: Any, dimensions: list[int]) -> dict[str, Any]:
        return dict(
            dimensions=dimensions,
            layers=[
                dict(
                    weight=m.weight.detach().numpy().tolist(), bias=m.bias.detach().numpy().tolist()
                )
                for m in net
                if isinstance(m, torch.nn.Linear)
            ],
        )

    result = dict(
        schema=SCHEMA,
        activation_ceiling="SIM_ONLY",
        learning_kind="per_frame_latent_gaussian_mc_ppo_clip_prototype",
        warm_start_model=warm,
        actor=export(actor, [134, 128, 128, 12]),
        critic=export(critic, [134, 128, 1]),
        physical_batch_hash=batch_hash,
        source_hash=hash_bytes(Path(__file__).read_bytes()),
        learning_receipt=dict(
            algorithm="PPO_CLIP_MC_TERMINAL",
            likelihood="unshielded Gaussian before tanh",
            gamma=1.0,
            gae_lambda=1.0,
            clip_ratio=0.2,
            learning_rate=learning_rate,
            requested_epochs=epochs,
            completed_optimizer_steps=steps,
            exact_mean_latent_kl=final_kl,
            kl_rejected_step=stopped,
            mean_policy_retention_coefficient=2.0,
            replay_teacher_frames=len(teachers),
            physical_batch_hash=batch_hash,
            physical_rollout_count=len(set(groups.tolist())),
            frame_sample_count=n,
            torch_version=torch.__version__,
            numpy_version=np.__version__,
            training_device="cpu",
            retention_guaranteed=False,
            optimizer_seed=20261001317,
        ),
        **dict.fromkeys(AUTHORITY_FLAGS, False),
    )
    result["model_hash"] = hash_json(result)
    validate_model(result)
    return result
