"""Frozen neural motor encoder with independently protected phase PPO heads.

Full successful trajectories can exhaust one shared protection plane. Three
sensor-grounded phases partition anchors, not the training data or success
criteria. Every protected action remains unchanged locally. Independent physics
retention is still required. Critic fitting is cross-fitted by whole rollout.
"""

import copy
import re
from pathlib import Path
from typing import Any

import numpy as np
import rosclaw.growth.anchor_plane as plane_module
from rosclaw.growth.anchor_plane import AnchorProtectionPlane

from rosclaw_soccer.rsi import step_motor_network as warm_network
from rosclaw_soccer.rsi import step_motor_phase_context as phase_module
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json

SCHEMA = "soccer.rsi.protected_phase_step_actor_critic.v1"
FLAGS = (
    "physics_qualified",
    "online_rl_qualified",
    "promotion_authorized",
    "runtime_execution_authorized",
    "hardware_authorized",
    "fresh_holdout_open_authorized",
)


def latents(model: dict[str, Any], observations: Any) -> np.ndarray[Any, Any]:
    x = np.asarray(observations, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != 134 or not np.isfinite(x).all():
        raise ValueError("finite causal motor feature matrix required")
    base = model["base_model"]
    norm = np.clip((x - np.asarray(base["mean"])) / np.asarray(base["scale"]), -8, 8)
    hidden = norm
    for layer in base["actor"]["layers"][:-1]:
        hidden = np.tanh(hidden @ np.asarray(layer["weight"]).T + np.asarray(layer["bias"]))
    random = model["frozen_random_features"]
    new = np.tanh(norm @ np.asarray(random["weight"]).T + np.asarray(random["bias"]))
    return np.concatenate((norm, hidden, new, np.ones((len(x), 1))), axis=1)


def validate_model(model: dict[str, Any]) -> list[AnchorProtectionPlane]:
    if (
        model.get("schema") != SCHEMA
        or model.get("activation_ceiling") != "SIM_ONLY"
        or model.get("learning_kind") != "frozen_encoder_protected_phase_mc_ppo"
        or model.get("source_hash") != hash_bytes(Path(__file__).read_bytes())
        or model.get("core_plane_source_hash")
        != hash_bytes(Path(plane_module.__file__).read_bytes())
        or model.get("phase_source_hash") != hash_bytes(Path(phase_module.__file__).read_bytes())
        or model.get("head_cap_raw") != 0.05
        or model.get("protected_active_frames") != 540
        or not re.fullmatch(r"sha256:[0-9a-f]{64}", model.get("anchor_bank_hash", ""))
        or type(model.get("generation")) is not int
        or model.get("phase_contract")
        != dict(force_input="previous_completed_frame", threshold_n=1.0, follow_through_frames=20)
        or any(model.get(k) is not False for k in FLAGS)
        or model.get("model_hash")
        != hash_json({k: v for k, v in model.items() if k != "model_hash"})
    ):
        raise ValueError("unsealed SIM-only protected phase motor model")
    warm_network.validate_model(model["base_model"])
    random = model["frozen_random_features"]
    for name, shape in (("weight", (249, 134)), ("bias", (249,))):
        value = np.asarray(random[name])
        if value.shape != shape or not np.isfinite(value).all():
            raise ValueError("finite immutable encoder expansion required")
    for name, readout_shape in (("actor_readout", (3, 12, 512)), ("critic_readout", (3, 512))):
        value = np.asarray(model[name])
        if value.shape != readout_shape or not np.isfinite(value).all():
            raise ValueError("finite aligned protected phase readouts required")
    if len(model["anchor_planes"]) != 3:
        raise ValueError("three independently protected physical phase planes required")
    planes = [AnchorProtectionPlane.from_dict(p) for p in model["anchor_planes"]]
    if any(p.dimension != 512 or p.plastic_dimensions <= 0 for p in planes):
        raise ValueError("nonempty phase-local plasticity required")
    if sum(len(p.to_dict()["anchors"]) for p in planes) != 540:
        raise ValueError("every successful active frame requires a protection anchor")
    if model["generation"] == 0:
        if model["learning_receipt"] is not None or np.any(np.asarray(model["actor_readout"]) != 0):
            raise ValueError("initial protected actor must reproduce frozen mean exactly")
    else:
        receipt = model.get("learning_receipt")
        if (
            model["generation"] != 1
            or not isinstance(receipt, dict)
            or receipt.get("algorithm") != "PROTECTED_PHASE_PPO_CLIP_MC_TERMINAL"
            or not re.fullmatch(r"sha256:[0-9a-f]{64}", receipt.get("physical_batch_hash", ""))
            or receipt.get("frozen_base_encoder") is not True
            or receipt.get("actor_head_only") is not True
            or receipt.get("distributional_retention_guaranteed") is not False
            or receipt.get("promotion_authorized") is not False
            or receipt.get("hardware_authorized") is not False
            or not np.isfinite(receipt.get("exact_mean_latent_kl", float("nan")))
            or not 0 <= receipt["exact_mean_latent_kl"] <= 0.005
        ):
            raise ValueError("explicit bounded physical-learning receipt required")
    return planes


def initial_model(
    base: dict[str, Any],
    protected_observations: Any,
    protected_phases: Any,
    *,
    anchor_bank_hash: str,
) -> dict[str, Any]:
    warm_network.validate_model(base)
    x, phases = np.asarray(protected_observations), np.asarray(protected_phases)
    if (
        x.shape != (540, 134)
        or phases.shape != (540,)
        or phases.dtype.kind not in "iu"
        or set(phases.tolist()) != {0, 1, 2}
    ):
        raise ValueError(
            "all 540 active frames of two successful predecessor trajectories required"
        )
    rng = np.random.default_rng(20261001327)
    model = dict(
        schema=SCHEMA,
        activation_ceiling="SIM_ONLY",
        learning_kind="frozen_encoder_protected_phase_mc_ppo",
        base_model=copy.deepcopy(base),
        frozen_random_features=dict(
            weight=(rng.normal(size=(249, 134)) / np.sqrt(134)).tolist(),
            bias=(rng.normal(size=249) * 0.1).tolist(),
            seed=20261001327,
        ),
        actor_readout=np.zeros((3, 12, 512)).tolist(),
        critic_readout=np.zeros((3, 512)).tolist(),
        generation=0,
        learning_receipt=None,
        head_cap_raw=0.05,
        anchor_bank_hash=anchor_bank_hash,
        protected_active_frames=540,
        phase_contract=dict(
            force_input="previous_completed_frame", threshold_n=1.0, follow_through_frames=20
        ),
        source_hash=hash_bytes(Path(__file__).read_bytes()),
        core_plane_source_hash=hash_bytes(Path(plane_module.__file__).read_bytes()),
        phase_source_hash=hash_bytes(Path(phase_module.__file__).read_bytes()),
        **dict.fromkeys(FLAGS, False),
    )
    phi = latents(model, x)
    model["anchor_planes"] = [AnchorProtectionPlane(phi[phases == p]).to_dict() for p in range(3)]
    model["model_hash"] = hash_json(model)
    validate_model(model)
    return model


def fit_update(model: dict[str, Any], arrays: Any, *, batch_hash: str) -> dict[str, Any]:
    import torch

    planes = validate_model(model)
    if model["generation"] != 0:
        raise ValueError("first protected learning round requires zero-head parent")
    x = np.asarray(arrays["observation"])
    phases = np.asarray(arrays["phase_index"])
    z = np.asarray(arrays["latent_action"])
    density = np.asarray(arrays["old_log_probability"])
    returns = np.asarray(arrays["terminal_return"])
    std = np.asarray(arrays["std_raw"])
    groups = np.asarray(arrays["trajectory_index"])
    n = len(x)
    if (
        not 100 <= n <= 30000
        or x.shape != (n, 134)
        or z.shape != (n, 12)
        or any(a.shape != (n,) for a in (phases, density, returns, std, groups))
        or not all(np.isfinite(a).all() for a in (x, z, phases, density, returns, std, groups))
        or phases.dtype.kind not in "iu"
        or groups.dtype.kind not in "iu"
        or set(phases.tolist()) != {0, 1, 2}
        or np.any((std < 0.01) | (std > 0.15))
        or not batch_hash.startswith("sha256:")
    ):
        raise ValueError("complete finite causal physical rollout batch required")
    phi = latents(model, x)
    base = model["base_model"]
    norm = np.clip((x - np.asarray(base["mean"])) / np.asarray(base["scale"]), -8, 8)
    means = norm
    for i, layer in enumerate(base["actor"]["layers"]):
        means = means @ np.asarray(layer["weight"]).T + np.asarray(layer["bias"])
        if i < 2:
            means = np.tanh(means)
    reconstructed = np.sum(
        -0.5 * ((z - means) / std[:, None]) ** 2 - np.log(std[:, None]) - 0.5 * np.log(2 * np.pi),
        axis=1,
    )
    if not np.allclose(density, reconstructed, atol=1e-8, rtol=0):
        raise ValueError("physical latent likelihood differs from frozen predecessor")
    if any(
        not np.all(returns[groups == g] == returns[groups == g][0]) for g in set(groups.tolist())
    ):
        raise ValueError("one terminal Monte Carlo return per physical rollout required")
    projected = np.stack([planes[int(p)].project(row) for p, row in zip(phases, phi, strict=True)])
    return_mean = base["critic_return_mean"]
    return_scale = base["critic_return_scale"]
    targets = (returns - return_mean) / return_scale
    value = np.zeros(n)
    final_critic = np.zeros((3, 512))
    ridge = 0.01

    def regression(ids: Any) -> np.ndarray[Any, Any]:
        design = phi[ids]
        if len(design) < 50:
            raise ValueError("sufficient whole-rollout cross-fit phase support required")
        return np.linalg.solve(design.T @ design + ridge * np.eye(512), design.T @ targets[ids])

    folds = groups % 4
    for p in range(3):
        for fold in range(4):
            train = (phases == p) & (folds != fold)
            test = (phases == p) & (folds == fold)
            value[test] = phi[test] @ regression(train)
        final_critic[p] = regression(phases == p)
    advantage = targets - value
    advantage = (advantage - advantage.mean()) / max(float(advantage.std()), 1e-6)
    torch.set_num_threads(4)
    torch.manual_seed(20261001327)
    torch.use_deterministic_algorithms(True)
    head = torch.nn.Parameter(torch.zeros((3, 12, 512), dtype=torch.float32))
    optimizer = torch.optim.Adam([head], lr=1e-3)
    inp = torch.tensor(projected, dtype=torch.float32)
    phase_ids = torch.tensor(phases, dtype=torch.int64)
    old_means = torch.tensor(means, dtype=torch.float32)
    actions = torch.tensor(z, dtype=torch.float32)
    noise = torch.tensor(std[:, None], dtype=torch.float32)
    advantages = torch.tensor(advantage, dtype=torch.float32)
    old_density = (
        -0.5 * ((actions - old_means) / noise).square() - noise.log() - 0.5 * np.log(2 * np.pi)
    ).sum(dim=1)

    def new_mean(ids: Any) -> Any:
        return old_means[ids] + 0.05 * torch.tanh(
            torch.einsum("noi,ni->no", head[phase_ids[ids]], inp[ids])
        )

    rng = np.random.default_rng(20261001327)
    all_ids = torch.arange(n)
    stopped = False
    steps = 0
    final_kl = 0.0
    for _ in range(4):
        for index in np.array_split(rng.permutation(n), max(1, (n + 1023) // 1024)):
            ids = torch.tensor(index, dtype=torch.int64)
            logp = (
                -0.5 * ((actions[ids] - new_mean(ids)) / noise[ids]).square()
                - noise[ids].log()
                - 0.5 * np.log(2 * np.pi)
            ).sum(dim=1)
            ratio = torch.exp(logp - old_density[ids])
            loss: Any = (
                -torch.minimum(
                    ratio * advantages[ids], ratio.clamp(0.8, 1.2) * advantages[ids]
                ).mean()
                + 1e-4 * head.square().mean()
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_([head], 1.0)
            previous = head.detach().clone()
            optimizer.step()
            with torch.no_grad():
                final_kl = float(
                    ((new_mean(all_ids) - old_means).square() / (2 * noise.square()))
                    .sum(dim=1)
                    .mean()
                )
            if final_kl > 0.005:
                with torch.no_grad():
                    head.copy_(previous)
                    final_kl = float(
                        ((new_mean(all_ids) - old_means).square() / (2 * noise.square()))
                        .sum(dim=1)
                        .mean()
                    )
                stopped = True
                break
            steps += 1
        if stopped:
            break
    result = {k: copy.deepcopy(v) for k, v in model.items() if k != "model_hash"}
    result["actor_readout"] = head.detach().numpy().astype(float).tolist()
    result["critic_readout"] = final_critic.tolist()
    result["generation"] = 1
    result["parent_model_hash"] = model["model_hash"]
    result["learning_receipt"] = dict(
        algorithm="PROTECTED_PHASE_PPO_CLIP_MC_TERMINAL",
        physical_batch_hash=batch_hash,
        gamma=1.0,
        gae_lambda=1.0,
        likelihood="unshielded Gaussian before tanh",
        frozen_base_encoder=True,
        actor_head_only=True,
        critic_crossfit_unit="whole_rollout",
        critic_crossfit_folds=4,
        critic_ridge=ridge,
        learning_rate=1e-3,
        requested_epochs=4,
        completed_optimizer_steps=steps,
        exact_mean_latent_kl=final_kl,
        kl_rejected_step=stopped,
        physical_rollout_count=len(set(groups.tolist())),
        frame_sample_count=n,
        protected_phase_frames=[len(p.to_dict()["anchors"]) for p in planes],
        plastic_dimensions=[p.plastic_dimensions for p in planes],
        local_protected_output_guarantee=True,
        distributional_retention_guaranteed=False,
        promotion_authorized=False,
        hardware_authorized=False,
    )
    result["model_hash"] = hash_json(result)
    validate_model(result)
    return result
