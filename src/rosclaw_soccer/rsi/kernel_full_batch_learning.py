"""Full-batch PPO with backtracking; frozen motor inference stays unchanged.

Consumed-only optimizer comparison, not a promotion or a safety guarantee.
Adam proposes a direction; penalized full-batch loss and cumulative Gaussian
KL decide whether/how much of that direction is accepted. No early minibatch
can alone consume the complete budget before the other courses contribute.
"""

import copy
import re
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.rsi.kernel_guarded_step_network import latents, validate_model, warm_means
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json


def fit_update(model: dict[str, Any], arrays: Any, *, batch_hash: str) -> dict[str, Any]:
    import torch

    guard = validate_model(model)
    x = np.asarray(arrays["observation"])
    phase = np.asarray(arrays["phase_index"])
    z = np.asarray(arrays["latent_action"])
    old_logp = np.asarray(arrays["old_log_probability"])
    returns = np.asarray(arrays["terminal_return"])
    std = np.asarray(arrays["std_raw"])
    groups = np.asarray(arrays["trajectory_index"])
    n = len(x)
    if (
        model["generation"] >= 32
        or not 100 <= n <= 200000
        or x.shape != (n, 134)
        or z.shape != (n, 12)
        or any(v.shape != (n,) for v in (phase, old_logp, returns, std, groups))
        or not all(np.isfinite(v).all() for v in (x, phase, z, old_logp, returns, std, groups))
        or phase.dtype.kind not in "iu"
        or groups.dtype.kind not in "iu"
        or set(phase.tolist()) != {0, 1, 2}
        or np.any(groups < 0)
        or np.any((std < 0.01) | (std > 0.15))
        or not re.fullmatch(r"sha256:[0-9a-f]{64}", batch_hash)
    ):
        raise ValueError("complete finite actual on-policy batch required")
    phi = latents(model, x)
    gates = guard.gates(phi[:, :134])
    parent_head = np.asarray(model["actor_readout"])
    base_mu = warm_means(model, phi)
    old_mu = base_mu + 0.05 * gates[:, None] * np.tanh(
        np.einsum("noi,ni->no", parent_head[phase], phi)
    )
    density = np.sum(
        -0.5 * ((z - old_mu) / std[:, None]) ** 2 - np.log(std[:, None]) - 0.5 * np.log(2 * np.pi),
        axis=1,
    )
    if not np.allclose(density, old_logp, atol=1e-8, rtol=0):
        raise ValueError("data is not the actual current parent policy")
    if any(
        not np.all(returns[groups == g] == returns[groups == g][0]) for g in set(groups.tolist())
    ):
        raise ValueError("one measured terminal return per physical trajectory required")
    warm = model["encoder"]["base_model"]
    targets = (returns - warm["critic_return_mean"]) / warm["critic_return_scale"]
    value = np.zeros(n)
    critic = np.zeros((3, 512))

    def regression(ids: Any) -> Any:
        design = phi[ids]
        if len(design) < 50:
            raise ValueError("whole-trajectory cross-fit support too small")
        return np.linalg.solve(design.T @ design + 0.01 * np.eye(512), design.T @ targets[ids])

    for p in range(3):
        for fold in range(4):
            test = (phase == p) & (groups % 4 == fold)
            value[test] = phi[test] @ regression((phase == p) & (groups % 4 != fold))
        critic[p] = regression(phase == p)
    advantage = targets - value
    advantage = (advantage - advantage.mean()) / max(float(advantage.std()), 1e-6)
    torch.set_num_threads(4)
    torch.manual_seed(202610335)
    torch.use_deterministic_algorithms(True)
    head = torch.nn.Parameter(torch.tensor(parent_head, dtype=torch.float32))
    optimizer = torch.optim.Adam([head], lr=1e-4)
    inp = torch.tensor(phi, dtype=torch.float32)
    base = torch.tensor(base_mu, dtype=torch.float32)
    noise = torch.tensor(std[:, None], dtype=torch.float32)
    action = torch.tensor(z, dtype=torch.float32)
    weights = torch.tensor(advantage, dtype=torch.float32)
    gate = torch.tensor(gates[:, None], dtype=torch.float32)
    ids = [torch.tensor(np.flatnonzero(phase == p), dtype=torch.int64) for p in range(3)]

    def mean() -> Any:
        # Do not materialize a [frames,12,512] selected-head tensor.
        residual = torch.zeros((n, 12), dtype=torch.float32)
        for p in range(3):
            residual = residual.index_copy(0, ids[p], inp[ids[p]] @ head[p].T)
        return base + 0.05 * gate * torch.tanh(residual)

    with torch.no_grad():
        original = mean().clone()
        logp0 = (
            -0.5 * ((action - original) / noise).square() - noise.log() - 0.5 * np.log(2 * np.pi)
        ).sum(dim=1)

    def objective() -> tuple[Any, Any]:
        mu = mean()
        logp = (
            -0.5 * ((action - mu) / noise).square() - noise.log() - 0.5 * np.log(2 * np.pi)
        ).sum(dim=1)
        ratio = torch.exp(logp - logp0)
        kl = ((mu - original).square() / (2 * noise.square())).sum(dim=1).mean()
        loss = -torch.minimum(ratio * weights, ratio.clamp(0.8, 1.2) * weights).mean()
        return loss + 10 * kl + 1e-4 * head.square().mean(), kl

    steps, reductions = 0, 0
    initial_loss = float(objective()[0].detach())
    losses = [initial_loss]
    for _ in range(40):
        optimizer.zero_grad()
        loss, _ = objective()
        if not torch.isfinite(loss):
            raise ValueError("nonfinite full-batch policy gradient")
        loss.backward()
        torch.nn.utils.clip_grad_norm_([head], 1.0)
        previous = head.detach().clone()
        optimizer.step()
        direction = head.detach().clone() - previous
        accepted = False
        for reduction in range(13):
            with torch.no_grad():
                head.copy_(previous + (0.5**reduction) * direction)
                trial_loss, trial_kl = objective()
                if (
                    torch.isfinite(trial_loss)
                    and float(trial_kl) <= 0.0049
                    and float(trial_loss) < losses[-1] - 1e-9
                ):
                    accepted = True
                    reductions += reduction
                    losses.append(float(trial_loss))
                    break
        if not accepted:
            with torch.no_grad():
                head.copy_(previous)
            break
        steps += 1
    learned = head.detach().numpy().astype(float)
    if not steps or np.array_equal(learned, parent_head):
        raise ValueError("optimizer did not produce a learned candidate")
    final_mu = base_mu + 0.05 * gates[:, None] * np.tanh(
        np.einsum("noi,ni->no", learned[phase], phi)
    )
    kl = float(np.mean(np.sum((final_mu - old_mu) ** 2 / (2 * std[:, None] ** 2), axis=1)))
    if not np.isfinite(kl) or kl > 0.005:
        raise ValueError("sealed float64 inference exceeded actual cumulative KL budget")
    result = {k: copy.deepcopy(v) for k, v in model.items() if k != "model_hash"}
    result.update(
        actor_readout=learned.tolist(),
        critic_readout=critic.tolist(),
        generation=model["generation"] + 1,
        parent_model_hash=model["model_hash"],
    )
    result["learning_receipt"] = dict(
        algorithm="KERNEL_GUARDED_PPO_CLIP_MC_TERMINAL",
        optimizer="FULL_BATCH_ADAM_DIRECTION_BACKTRACKING",
        optimizer_source_hash=hash_bytes(Path(__file__).read_bytes()),
        physical_batch_hash=batch_hash,
        frozen_encoder=True,
        gamma=1.0,
        gae_lambda=1.0,
        critic_crossfit_unit="whole_rollout",
        critic_crossfit_folds=4,
        critic_ridge=0.01,
        learning_rate=1e-4,
        requested_epochs=40,
        completed_optimizer_steps=steps,
        backtracking_reductions=reductions,
        full_batch_loss_history=losses,
        kl_penalty=10.0,
        exact_mean_latent_kl=kl,
        physical_rollout_count=len(set(groups.tolist())),
        frame_sample_count=n,
        protected_frames=model["protected_frames"],
        local_protected_output_guarantee=True,
        distributional_retention_guaranteed=False,
        promotion_authorized=False,
        hardware_authorized=False,
    )
    result["model_hash"] = hash_json(result)
    validate_model(result)
    return result
