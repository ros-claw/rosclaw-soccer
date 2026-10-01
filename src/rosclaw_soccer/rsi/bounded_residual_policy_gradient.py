"""Task-neutral, bounded full-batch IID Gaussian residual PPO optimization.

Numeric arrays only. No simulator, trajectory labels, policy activation, or
hardware authority. The caller must audit data provenance and final inference.
Kept separate from historical artifact-bound optimizers for paired experiments.
"""

from typing import Any

import numpy as np


def optimize_residual(
    layers: list[dict[str, Any]],
    context: Any,
    baseline: Any,
    gates: Any,
    actions: Any,
    std: Any,
    advantage: Any,
    *,
    raw_cap: float = 0.05,
    learning_rate: float = 1e-4,
    steps: int = 160,
    seed: int = 202610343,
) -> dict[str, Any]:
    import torch

    x, base, gate, desired, noise, importance = [
        np.asarray(v, dtype=np.float64) for v in (context, baseline, gates, actions, std, advantage)
    ]
    n = len(x)
    if (
        x.ndim != 2
        or not 100 <= n <= 200000
        or not 1 <= x.shape[1] <= 512
        or desired.ndim != 2
        or desired.shape[0] != n
        or not 1 <= desired.shape[1] <= 512
        or base.shape != desired.shape
        or any(v.shape != (n,) for v in (gate, noise, importance))
        or not all(np.isfinite(v).all() for v in (x, base, gate, desired, noise, importance))
        or max(float(np.max(np.abs(v))) for v in (x, base, desired, importance)) > 1e6
        or np.any((gate < 0) | (gate > 1))
        or np.any((noise < 0.01) | (noise > 0.15))
        or type(raw_cap) not in (float, int)
        or not np.isfinite(raw_cap)
        or not 0.001 <= raw_cap <= 1
        or type(learning_rate) not in (float, int)
        or not np.isfinite(learning_rate)
        or not 1e-5 <= learning_rate <= 1e-3
        or type(steps) is not int
        or not 1 <= steps <= 160
        or type(seed) is not int
        or not 0 <= seed < 2**32
        or not 1 <= len(layers) <= 4
    ):
        raise ValueError("finite bounded aligned residual PPO inputs required")
    frozen_layers = []
    input_dimension = x.shape[1]
    for layer in layers:
        weight, bias = np.asarray(layer["weight"]), np.asarray(layer["bias"])
        if (
            weight.ndim != 2
            or weight.shape[1] != input_dimension
            or not 1 <= weight.shape[0] <= 512
            or bias.shape != (weight.shape[0],)
            or not np.isfinite(weight).all()
            or not np.isfinite(bias).all()
            or max(float(np.max(np.abs(v))) for v in (weight, bias)) > 1e6
        ):
            raise ValueError("finite aligned bounded residual layers required")
        frozen_layers.append((weight, bias))
        input_dimension = weight.shape[0]
    if input_dimension != desired.shape[1]:
        raise ValueError("residual output dimension does not match actions")
    torch.set_num_threads(4)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    weights = [torch.nn.Parameter(torch.tensor(w, dtype=torch.float32)) for w, _ in frozen_layers]
    biases = [torch.nn.Parameter(torch.tensor(b, dtype=torch.float32)) for _, b in frozen_layers]
    parameters = weights + biases
    optimizer = torch.optim.Adam(parameters, lr=learning_rate)
    inp = torch.tensor(x, dtype=torch.float32)
    base_t = torch.tensor(base, dtype=torch.float32)
    gate_t = torch.tensor(gate[:, None], dtype=torch.float32)
    noise_t = torch.tensor(noise[:, None], dtype=torch.float32)
    desired_t = torch.tensor(desired, dtype=torch.float32)
    importance_t = torch.tensor(importance, dtype=torch.float32)

    def mean() -> Any:
        hidden = inp
        for w, b in zip(weights, biases, strict=True):
            hidden = torch.tanh(hidden @ w.T + b)
        return base_t + raw_cap * gate_t * hidden

    with torch.no_grad():
        original = mean().clone()
        logp0 = (
            -0.5 * ((desired_t - original) / noise_t).square()
            - noise_t.log()
            - 0.5 * np.log(2 * np.pi)
        ).sum(dim=1)

    def objective() -> tuple[Any, Any]:
        mu = mean()
        logp = (
            -0.5 * ((desired_t - mu) / noise_t).square() - noise_t.log() - 0.5 * np.log(2 * np.pi)
        ).sum(dim=1)
        ratio = torch.exp(logp - logp0)
        kl = ((mu - original).square() / (2 * noise_t.square())).sum(dim=1).mean()
        loss = -torch.minimum(ratio * importance_t, ratio.clamp(0.8, 1.2) * importance_t).mean()
        return loss + 10 * kl, kl

    initial, _ = objective()
    if not torch.isfinite(initial):
        raise ValueError("nonfinite initial residual objective")
    history = [float(initial.detach())]
    reductions = 0
    for _ in range(steps):
        optimizer.zero_grad()
        loss, _ = objective()
        if not torch.isfinite(loss):
            raise ValueError("nonfinite residual policy gradient")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        previous = [p.detach().clone() for p in parameters]
        optimizer.step()
        directions = [p.detach().clone() - old for p, old in zip(parameters, previous, strict=True)]
        accepted = False
        for reduction in range(13):
            with torch.no_grad():
                for parameter, old, d in zip(parameters, previous, directions, strict=True):
                    parameter.copy_(old + (0.5**reduction) * d)
                trial, kl = objective()
                if (
                    torch.isfinite(trial)
                    and float(kl) <= 0.0049
                    and float(trial) < history[-1] - 1e-9
                ):
                    history.append(float(trial))
                    reductions += reduction
                    accepted = True
                    break
        if not accepted:
            with torch.no_grad():
                for parameter, old in zip(parameters, previous, strict=True):
                    parameter.copy_(old)
            break
    if len(history) == 1:
        raise ValueError("no actual residual network learning")
    return dict(
        layers=[
            dict(
                weight=w.detach().numpy().astype(float).tolist(),
                bias=b.detach().numpy().astype(float).tolist(),
            )
            for w, b in zip(weights, biases, strict=True)
        ],
        full_batch_loss_history=history,
        backtracking_reductions=reductions,
        completed_optimizer_steps=len(history) - 1,
        torch_mean_latent_kl=float(objective()[1].detach()),
        learning_rate=float(learning_rate),
        raw_cap=float(raw_cap),
        promotion_authorized=False,
        hardware_authorized=False,
    )
