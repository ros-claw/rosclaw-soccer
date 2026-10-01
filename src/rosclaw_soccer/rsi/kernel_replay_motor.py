"""AWR-inspired consumed replay experiment, distinct from the frozen PPO codec.

Only the original audited warm behavior is accepted by this first experiment.
No general off-policy correction, fresh qualification or physical retention is
claimed. Inference reuses the frozen kernel decoder and its motor envelope.
"""

import copy
import re
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.rsi.contact_motor_phase import make_policy
from rosclaw_soccer.rsi.kernel_guarded_step_execution import CompiledKernelStepMotor
from rosclaw_soccer.rsi.kernel_guarded_step_execution import make_preview as kernel_preview
from rosclaw_soccer.rsi.kernel_guarded_step_network import FLAGS, latents, warm_means
from rosclaw_soccer.rsi.kernel_guarded_step_network import validate_model as kernel_validate
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json

SCHEMA = "soccer.rsi.kernel_replay_motor.v1"


def validate_model(model: dict[str, Any]) -> None:
    if (
        model.get("schema") != SCHEMA
        or model.get("activation_ceiling") != "SIM_ONLY"
        or model.get("source_hash") != hash_bytes(Path(__file__).read_bytes())
        or any(model.get(k) is not False for k in FLAGS)
        or model.get("model_hash")
        != hash_json({k: v for k, v in model.items() if k != "model_hash"})
    ):
        raise ValueError("sealed SIM-only replay motor required")
    parent = model["frozen_parent"]
    kernel_validate(parent)
    if parent["generation"] != 0 or model["parent_model_hash"] != parent["model_hash"]:
        raise ValueError("first replay experiment must retain the original warm parent")
    receipt = model["learning_receipt"]
    if (
        receipt.get("algorithm") != "AWR_INSPIRED_TERMINAL_GAUSSIAN_REGRESSION"
        or receipt.get("frozen_encoder") is not True
        or receipt.get("distributional_retention_guaranteed") is not False
        or any(receipt.get(k) is not False for k in ("promotion_authorized", "hardware_authorized"))
        or not re.fullmatch(r"sha256:[0-9a-f]{64}", receipt.get("physical_batch_hash", ""))
        or type(receipt.get("completed_optimizer_steps")) is not int
        or not 1 <= receipt["completed_optimizer_steps"] <= 160
        or type(receipt.get("exact_mean_latent_kl")) not in (int, float)
        or not np.isfinite(receipt["exact_mean_latent_kl"])
        or not 0 <= receipt["exact_mean_latent_kl"] <= 0.005
    ):
        raise ValueError("actual bounded replay-learning receipt required")
    for key, shape in (("actor_readout", (3, 12, 512)), ("critic_readout", (3, 512))):
        value = np.asarray(model[key])
        if value.shape != shape or not np.isfinite(value).all():
            raise ValueError("finite aligned replay readouts required")


def fit_update(parent: dict[str, Any], arrays: Any, *, batch_hash: str) -> dict[str, Any]:
    import torch

    guard = kernel_validate(parent)
    x, phase, action, old_logp, returns, std, groups = [
        np.asarray(arrays[k])
        for k in (
            "observation",
            "phase_index",
            "latent_action",
            "old_log_probability",
            "terminal_return",
            "std_raw",
            "trajectory_index",
        )
    ]
    n = len(x)
    if (
        parent["generation"] != 0
        or not 100 <= n <= 200000
        or x.shape != (n, 134)
        or action.shape != (n, 12)
        or any(a.shape != (n,) for a in (phase, old_logp, returns, std, groups))
        or not all(np.isfinite(a).all() for a in (x, phase, action, old_logp, returns, std, groups))
        or phase.dtype.kind not in "iu"
        or groups.dtype.kind not in "iu"
        or set(phase.tolist()) != {0, 1, 2}
        or np.any(groups < 0)
        or np.any((std < 0.01) | (std > 0.15))
        or not re.fullmatch(r"sha256:[0-9a-f]{64}", batch_hash)
    ):
        raise ValueError("complete audited warm-behavior replay batch required")
    phi = latents(parent, x)
    base = warm_means(parent, phi)
    gates = guard.gates(phi[:, :134])
    density = np.sum(
        -0.5 * ((action - base) / std[:, None]) ** 2
        - np.log(std[:, None])
        - 0.5 * np.log(2 * np.pi),
        axis=1,
    )
    if not np.allclose(density, old_logp, atol=1e-8, rtol=0):
        raise ValueError("replay density is not the declared warm behavior")
    if any(not np.all(returns[groups == g] == returns[groups == g][0]) for g in set(groups)):
        raise ValueError("one actual terminal return per trajectory required")
    warm = parent["encoder"]["base_model"]
    targets = (returns - warm["critic_return_mean"]) / warm["critic_return_scale"]
    value = np.zeros(n)
    critic = np.zeros((3, 512))

    def regression(mask: Any) -> Any:
        design = phi[mask]
        if len(design) < 50:
            raise ValueError("whole-trajectory cross-fit support too small")
        return np.linalg.solve(design.T @ design + 0.01 * np.eye(512), design.T @ targets[mask])

    for p in range(3):
        for fold in range(4):
            test = (phase == p) & (groups % 4 == fold)
            value[test] = phi[test] @ regression((phase == p) & (groups % 4 != fold))
        critic[p] = regression(phase == p)
    advantage = targets - value
    advantage = (advantage - advantage.mean()) / max(float(advantage.std()), 1e-6)
    weights = np.exp(np.clip(advantage, -8, np.log(20)))
    weights /= weights.mean()
    torch.set_num_threads(4)
    torch.manual_seed(202610340)
    torch.use_deterministic_algorithms(True)
    head = torch.nn.Parameter(torch.zeros((3, 12, 512), dtype=torch.float32))
    optimizer = torch.optim.Adam([head], lr=1e-4)
    inp = torch.tensor(phi, dtype=torch.float32)
    nominal = torch.tensor(base, dtype=torch.float32)
    goal = torch.tensor(action, dtype=torch.float32)
    noise = torch.tensor(std[:, None], dtype=torch.float32)
    gate = torch.tensor(gates[:, None], dtype=torch.float32)
    weight = torch.tensor(weights, dtype=torch.float32)
    ids = [torch.tensor(np.flatnonzero(phase == p), dtype=torch.int64) for p in range(3)]

    def objective() -> tuple[Any, Any]:
        residual = torch.zeros((n, 12), dtype=torch.float32)
        for p in range(3):
            residual = residual.index_copy(0, ids[p], inp[ids[p]] @ head[p].T)
        mean = nominal + 0.05 * gate * torch.tanh(residual)
        kl = ((mean - nominal).square() / (2 * noise.square())).sum(dim=1).mean()
        loss = (weight * ((mean - goal).square() / (2 * noise.square())).sum(dim=1)).mean()
        return loss + 10 * kl + 1e-4 * head.square().mean(), kl

    history = [float(objective()[0].detach())]
    reductions = 0
    for _ in range(160):
        optimizer.zero_grad()
        loss, _ = objective()
        if not torch.isfinite(loss):
            raise ValueError("nonfinite weighted replay regression")
        loss.backward()
        torch.nn.utils.clip_grad_norm_([head], 1.0)
        previous = head.detach().clone()
        optimizer.step()
        direction = head.detach().clone() - previous
        accepted = False
        for reduction in range(13):
            with torch.no_grad():
                head.copy_(previous + (0.5**reduction) * direction)
                trial, kl = objective()
                if (
                    torch.isfinite(trial)
                    and float(kl) <= 0.0049
                    and float(trial) < history[-1] - 1e-7
                ):
                    history.append(float(trial))
                    reductions += reduction
                    accepted = True
                    break
        if not accepted:
            with torch.no_grad():
                head.copy_(previous)
            break
    learned = head.detach().numpy().astype(float)
    if len(history) == 1 or not np.any(learned):
        raise ValueError("replay optimizer produced no learned candidate")
    mean = base + 0.05 * gates[:, None] * np.tanh(np.einsum("noi,ni->no", learned[phase], phi))
    kl = float(np.mean(np.sum((mean - base) ** 2 / (2 * std[:, None] ** 2), axis=1)))
    model = dict(
        schema=SCHEMA,
        activation_ceiling="SIM_ONLY",
        frozen_parent=copy.deepcopy(parent),
        parent_model_hash=parent["model_hash"],
        actor_readout=learned.tolist(),
        critic_readout=critic.tolist(),
        source_hash=hash_bytes(Path(__file__).read_bytes()),
        learning_receipt=dict(
            algorithm="AWR_INSPIRED_TERMINAL_GAUSSIAN_REGRESSION",
            physical_batch_hash=batch_hash,
            completed_optimizer_steps=len(history) - 1,
            full_batch_loss_history=history,
            backtracking_reductions=reductions,
            exact_mean_latent_kl=kl,
            frozen_encoder=True,
            distributional_retention_guaranteed=False,
            physical_rollout_count=len(set(groups.tolist())),
            frame_sample_count=n,
            critic_crossfit_unit="whole_rollout",
            critic_crossfit_folds=4,
            advantage_standardized=True,
            temperature=1.0,
            unnormalized_weight_cap=20.0,
            gamma=1.0,
            kl_penalty=10.0,
            learning_rate=1e-4,
            promotion_authorized=False,
            hardware_authorized=False,
        ),
        **dict.fromkeys(FLAGS, False),
    )
    model["model_hash"] = hash_json(model)
    validate_model(model)
    return model


def make_preview(model: dict[str, Any]) -> dict[str, Any]:
    validate_model(model)
    policy = make_policy(np.zeros((3, 12)), 0.25, model["model_hash"])
    policy["execution_profile"] = "causal_per_frame_neural_residual"
    policy["step_motor_proof"] = dict(
        schema="soccer.rsi.kernel_replay_preview.v1",
        model=model,
        decision_start_frame=30,
        nominal_target_is_pre_motor=True,
        force_input="previous_completed_frame",
        execution_source_hash=hash_bytes(Path(__file__).read_bytes()),
        qualification="UNQUALIFIED_SIM_COUNTERFACTUAL",
        promotion_authorized=False,
    )
    policy["replay_motor_proof"] = dict(
        parent_model_hash=model["parent_model_hash"],
        algorithm=model["learning_receipt"]["algorithm"],
        promotion_authorized=False,
    )
    policy.pop("policy_hash")
    policy["policy_hash"] = hash_json(policy)
    return policy


class CompiledReplayStepMotor(CompiledKernelStepMotor):
    def __init__(self, policy: dict[str, Any]) -> None:
        model = copy.deepcopy(policy["step_motor_proof"]["model"])
        if make_preview(model) != policy:
            raise ValueError("replay inference commitment changed")
        super().__init__(kernel_preview(model["frozen_parent"]))
        self._head = np.asarray(model["actor_readout"], dtype=np.float64)
        self._head.flags.writeable = False
        self._policy_hash = policy["policy_hash"]
