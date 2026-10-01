"""Bounded per-frame actor/critic learning with whole successful-state memory.

Only frozen states are protected exactly, not their whole linear span. Physical
retention, CPU transfer and unseen evaluation remain mandatory. Success labels
build the training anchor bank; none enter the running motor policy.
"""

import copy
import re
from pathlib import Path
from typing import Any

import numpy as np
import rosclaw.growth.anchor_kernel as kernel_module
from rosclaw.growth.anchor_kernel import AnchorKernelGuard

from rosclaw_soccer.rsi import protected_phase_step_network as encoder_module
from rosclaw_soccer.rsi import step_motor_phase_context as phase_module
from rosclaw_soccer.rsi.step_motor_network import validate_model as validate_warm
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json

SCHEMA = "soccer.rsi.kernel_guarded_step_actor_critic.v1"
FLAGS = encoder_module.FLAGS


def latents(model: dict[str, Any], x: Any) -> np.ndarray[Any, Any]:
    return encoder_module.latents(model["encoder"], x)


def validate_model(model: dict[str, Any]) -> AnchorKernelGuard:
    if (
        model.get("schema") != SCHEMA
        or model.get("activation_ceiling") != "SIM_ONLY"
        or model.get("source_hash") != hash_bytes(Path(__file__).read_bytes())
        or model.get("core_kernel_source_hash")
        != hash_bytes(Path(kernel_module.__file__).read_bytes())
        or model.get("encoder_source_hash")
        != hash_bytes(Path(encoder_module.__file__).read_bytes())
        or model.get("phase_source_hash") != hash_bytes(Path(phase_module.__file__).read_bytes())
        or model.get("head_cap_raw") != 0.05
        or not re.fullmatch(r"sha256:[0-9a-f]{64}", model.get("anchor_bank_hash", ""))
        or any(model.get(k) is not False for k in FLAGS)
        or type(model.get("generation")) is not int
        or not 0 <= model["generation"] <= 32
        or model.get("model_hash")
        != hash_json({k: v for k, v in model.items() if k != "model_hash"})
    ):
        raise ValueError("unsealed SIM-only kernel-protected motor learner")
    validate_warm(model["encoder"]["base_model"])
    random = model["encoder"]["frozen_random_features"]
    for value, shape in (
        (random["weight"], (249, 134)),
        (random["bias"], (249,)),
        (model["actor_readout"], (3, 12, 512)),
        (model["critic_readout"], (3, 512)),
    ):
        a = np.asarray(value)
        if a.shape != shape or not np.isfinite(a).all():
            raise ValueError("finite aligned frozen encoder and plastic readouts required")
    guard = AnchorKernelGuard.from_dict(model["anchor_guard"])
    if guard.dimension != 134 or model["protected_frames"] != len(model["anchor_guard"]["anchors"]):
        raise ValueError("complete frozen successful-state guard required")
    if model["generation"] == 0:
        if model["learning_receipt"] is not None or np.any(np.asarray(model["actor_readout"]) != 0):
            raise ValueError("zero-head parent must reproduce warm policy")
    else:
        r = model.get("learning_receipt")
        if (
            not isinstance(r, dict)
            or r.get("algorithm") != "KERNEL_GUARDED_PPO_CLIP_MC_TERMINAL"
            or not re.fullmatch(r"sha256:[0-9a-f]{64}", r.get("physical_batch_hash", ""))
            or r.get("frozen_encoder") is not True
            or r.get("distributional_retention_guaranteed") is not False
            or r.get("promotion_authorized") is not False
            or r.get("hardware_authorized") is not False
            or not np.isfinite(r.get("exact_mean_latent_kl", float("nan")))
            or not 0 <= r["exact_mean_latent_kl"] <= 0.005
        ):
            raise ValueError("actual bounded physical-learning receipt required")
    return guard


def initial_model(
    encoder: dict[str, Any], anchors: Any, *, anchor_bank_hash: str
) -> dict[str, Any]:
    validate_warm(encoder["base_model"])
    x = np.asarray(anchors)
    if x.ndim != 2 or x.shape[1] != 134 or not 1 <= len(x) <= 32768:
        raise ValueError("all declared successful causal states required")
    template = copy.deepcopy(encoder)
    phi = encoder_module.latents(template, x)
    # Committed metric scale; never selected from candidate/fresh physics scores.
    guard = AnchorKernelGuard(phi[:, :134], bandwidth=0.1)
    model = dict(
        schema=SCHEMA,
        activation_ceiling="SIM_ONLY",
        encoder=template,
        actor_readout=np.zeros((3, 12, 512)).tolist(),
        critic_readout=np.zeros((3, 512)).tolist(),
        anchor_guard=guard.to_dict(),
        anchor_bank_hash=anchor_bank_hash,
        protected_frames=len(x),
        head_cap_raw=0.05,
        generation=0,
        learning_receipt=None,
        source_hash=hash_bytes(Path(__file__).read_bytes()),
        core_kernel_source_hash=hash_bytes(Path(kernel_module.__file__).read_bytes()),
        encoder_source_hash=hash_bytes(Path(encoder_module.__file__).read_bytes()),
        phase_source_hash=hash_bytes(Path(phase_module.__file__).read_bytes()),
        **dict.fromkeys(FLAGS, False),
    )
    model["model_hash"] = hash_json(model)
    validate_model(model)
    return model


def warm_means(model: dict[str, Any], phi: Any) -> np.ndarray[Any, Any]:
    last = model["encoder"]["base_model"]["actor"]["layers"][-1]
    return np.asarray(phi[:, 134:262] @ np.asarray(last["weight"]).T + np.asarray(last["bias"]))


def fit_update(model: dict[str, Any], arrays: Any, *, batch_hash: str) -> dict[str, Any]:
    import torch

    guard = validate_model(model)
    if model["generation"] >= 32:
        raise ValueError("explicit continual-generation capacity exhausted")
    x = np.asarray(arrays["observation"])
    phase = np.asarray(arrays["phase_index"])
    z = np.asarray(arrays["latent_action"])
    old_logp = np.asarray(arrays["old_log_probability"])
    returns = np.asarray(arrays["terminal_return"])
    std = np.asarray(arrays["std_raw"])
    groups = np.asarray(arrays["trajectory_index"])
    n = len(x)
    if (
        not 100 <= n <= 200000
        or x.shape != (n, 134)
        or z.shape != (n, 12)
        or any(v.shape != (n,) for v in (phase, old_logp, returns, std, groups))
        or not all(np.isfinite(v).all() for v in (x, phase, z, old_logp, returns, std, groups))
        or phase.dtype.kind not in "iu"
        or groups.dtype.kind not in "iu"
        or set(phase.tolist()) != {0, 1, 2}
        or np.any((std < 0.01) | (std > 0.15))
        or not re.fullmatch(r"sha256:[0-9a-f]{64}", batch_hash)
    ):
        raise ValueError("finite complete on-policy physical rollout batch required")
    phi = latents(model, x)
    gates = guard.gates(phi[:, :134])
    parent_head = np.asarray(model["actor_readout"])
    base_means = warm_means(model, phi)
    old_means = base_means + 0.05 * gates[:, None] * np.tanh(
        np.einsum("noi,ni->no", parent_head[phase], phi)
    )
    density = np.sum(
        -0.5 * ((z - old_means) / std[:, None]) ** 2
        - np.log(std[:, None])
        - 0.5 * np.log(2 * np.pi),
        axis=1,
    )
    if not np.allclose(density, old_logp, atol=1e-8, rtol=0):
        raise ValueError("physical likelihood is not the actual current parent policy")
    if any(
        not np.all(returns[groups == g] == returns[groups == g][0]) for g in set(groups.tolist())
    ):
        raise ValueError("one actual terminal MC return per rollout required")
    base = model["encoder"]["base_model"]
    targets = (returns - base["critic_return_mean"]) / base["critic_return_scale"]
    value = np.zeros(n)
    critic = np.zeros((3, 512))

    def regression(ids: Any) -> Any:
        design = phi[ids]
        if len(design) < 50:
            raise ValueError("whole-rollout cross-fit support too small")
        return np.linalg.solve(design.T @ design + 0.01 * np.eye(512), design.T @ targets[ids])

    for p in range(3):
        for f in range(4):
            test = (phase == p) & (groups % 4 == f)
            value[test] = phi[test] @ regression((phase == p) & (groups % 4 != f))
        critic[p] = regression(phase == p)
    advantage = targets - value
    advantage = (advantage - advantage.mean()) / max(float(advantage.std()), 1e-6)
    torch.set_num_threads(4)
    torch.manual_seed(20261001331)
    torch.use_deterministic_algorithms(True)
    head = torch.nn.Parameter(torch.tensor(parent_head, dtype=torch.float32))
    optimizer = torch.optim.Adam([head], lr=3e-4)
    inp = torch.tensor(phi, dtype=torch.float32)
    pids = torch.tensor(phase, dtype=torch.int64)
    gate = torch.tensor(gates[:, None], dtype=torch.float32)
    base_mu = torch.tensor(base_means, dtype=torch.float32)
    noise = torch.tensor(std[:, None], dtype=torch.float32)
    actions = torch.tensor(z, dtype=torch.float32)
    advantages = torch.tensor(advantage, dtype=torch.float32)

    def mean(ids: Any) -> Any:
        return base_mu[ids] + 0.05 * gate[ids] * torch.tanh(
            torch.einsum("noi,ni->no", head[pids[ids]], inp[ids])
        )

    all_ids = torch.arange(n)
    with torch.no_grad():
        old_mu = mean(all_ids).clone()
        logp0 = (
            -0.5 * ((actions - old_mu) / noise).square() - noise.log() - 0.5 * np.log(2 * np.pi)
        ).sum(dim=1)
    rng = np.random.default_rng(20261001331)
    steps, kl, rejected = 0, 0.0, False
    for _ in range(4):
        for index in np.array_split(rng.permutation(n), max(1, (n + 1023) // 1024)):
            ids = torch.tensor(index, dtype=torch.int64)
            logp = (
                -0.5 * ((actions[ids] - mean(ids)) / noise[ids]).square()
                - noise[ids].log()
                - 0.5 * np.log(2 * np.pi)
            ).sum(dim=1)
            ratio = torch.exp(logp - logp0[ids])
            loss: Any = -torch.minimum(
                ratio * advantages[ids], ratio.clamp(0.8, 1.2) * advantages[ids]
            ).mean()
            loss = loss + 1e-4 * head.square().mean()
            if not torch.isfinite(loss):
                raise ValueError("nonfinite policy gradient cannot produce a candidate")
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_([head], 1.0)
            previous = head.detach().clone()
            optimizer.step()
            with torch.no_grad():
                kl = float(
                    ((mean(all_ids) - old_mu).square() / (2 * noise.square())).sum(dim=1).mean()
                )
            if kl > 0.005:
                with torch.no_grad():
                    head.copy_(previous)
                    kl = float(
                        ((mean(all_ids) - old_mu).square() / (2 * noise.square())).sum(dim=1).mean()
                    )
                rejected = True
                break
            steps += 1
        if rejected:
            break
    result = {k: copy.deepcopy(v) for k, v in model.items() if k != "model_hash"}
    result.update(
        actor_readout=head.detach().numpy().astype(float).tolist(),
        critic_readout=critic.tolist(),
        generation=model["generation"] + 1,
        parent_model_hash=model["model_hash"],
    )
    result["learning_receipt"] = dict(
        algorithm="KERNEL_GUARDED_PPO_CLIP_MC_TERMINAL",
        physical_batch_hash=batch_hash,
        frozen_encoder=True,
        gamma=1.0,
        gae_lambda=1.0,
        critic_crossfit_unit="whole_rollout",
        critic_crossfit_folds=4,
        critic_ridge=0.01,
        learning_rate=3e-4,
        requested_epochs=4,
        completed_optimizer_steps=steps,
        exact_mean_latent_kl=kl,
        kl_rejected_step=rejected,
        physical_rollout_count=len(set(groups.tolist())),
        frame_sample_count=n,
        protected_frames=model["protected_frames"],
        fraction_zero_gate=float(np.mean(gates == 0)),
        local_protected_output_guarantee=True,
        distributional_retention_guaranteed=False,
        promotion_authorized=False,
        hardware_authorized=False,
    )
    result["model_hash"] = hash_json(result)
    validate_model(result)
    return result
