"""Persistent, CPU-float64 latent PPO actor/critic for SIM torque research.

This is a numeric optimizer, not a dataset authenticator or skill qualification.
The caller must bind every complete on-policy episode and retention anchor to
independently replayed physics. JSON checkpoints retain Adam moments and critic;
no pickle, simulator, hardware, or automatic policy activation is involved.
"""

import copy
import re
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.rsi import cerebellar_recovery_return as reward_module
from rosclaw_soccer.rsi import cerebellar_torque_policy as policy
from rosclaw_soccer.rsi.cerebellar_recovery_return import reward_contract
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json

SCHEMA = "soccer.rsi.cerebellar_actor_critic_checkpoint.v1"
CONFIG: dict[str, Any] = dict(
    algorithm="PPO_CLIP_MC_TERMINAL",
    likelihood="unprojected Gaussian latent before tanh",
    latent_std=0.05,
    gamma=1.0,
    return_scale=10.0,
    advantage_centered=False,
    bootstrap_after_deadline=False,
    learning_rate=3e-5,
    betas=[0.9, 0.999],
    eps=1e-8,
    clip_ratio=0.2,
    max_grad_norm=1.0,
    critic_loss_coefficient=0.5,
    anchor_kl_coefficient=0.1,
    maximum_rollout_mean_kl=0.0049,
    maximum_rollout_state_kl=0.1,
    maximum_anchor_mean_kl=0.001,
    maximum_anchor_state_kl=0.02,
)


def _hash(value: Any) -> bool:
    return type(value) is str and re.fullmatch(r"sha256:[0-9a-f]{64}", value) is not None


def _finite(value: Any, shape: tuple[int, ...], *, bound: float) -> np.ndarray:
    raw = np.asarray(value)
    if (
        raw.shape != shape
        or raw.dtype.kind not in "fi"
        or not np.isfinite(raw).all()
        or np.max(np.abs(raw)) > bound
    ):
        raise ValueError("finite aligned bounded actor/critic numeric arrays required")
    return np.asarray(raw, dtype=np.float64)


def _anchors(value: Any) -> np.ndarray:
    raw = np.asarray(value)
    if raw.ndim != 2 or not 100 <= len(raw) <= 24000 or raw.dtype != np.float64:
        raise ValueError("100..24000 float64 retention features required")
    raw = _finite(raw, (len(raw), 127), bound=10)
    if np.any((raw[:, -1] <= 0) | (raw[:, -1] > 1)):
        raise ValueError("retention features need their causal task clock")
    return raw


def _anchor_hash(value: np.ndarray) -> str:
    return hash_bytes(np.ascontiguousarray(value, dtype="<f8").tobytes())


def _parameters(layers: list[dict[str, Any]]) -> list[np.ndarray]:
    return [
        np.asarray(layer[key], dtype=np.float64) for layer in layers for key in ("weight", "bias")
    ]


def _seal(value: dict[str, Any]) -> dict[str, Any]:
    result = {k: v for k, v in value.items() if k != "checkpoint_hash"}
    result["checkpoint_hash"] = hash_json(result)
    validate_checkpoint(result)
    return result


def validate_checkpoint(value: dict[str, Any]) -> None:
    if (
        type(value) is not dict
        or value.get("schema") != SCHEMA
        or value.get("config") != CONFIG
        or value.get("training_reward_contract_hash") != reward_contract()["contract_hash"]
        or value.get("activation_ceiling") != "SIM_ONLY"
        or any(
            value.get(k) is not False
            for k in (
                "hardware_authorized",
                "promotion_authorized",
                "runtime_execution_authorized",
                "teacher_qualified",
                "physics_qualified",
                "fresh_holdout_open_authorized",
            )
        )
        or value.get("source_hashes")
        != dict(
            optimizer=hash_bytes(Path(__file__).read_bytes()),
            policy=hash_bytes(Path(policy.__file__).read_bytes()),
            reward=hash_bytes(Path(reward_module.__file__).read_bytes()),
        )
        or not _hash(value.get("anchor_features_hash"))
        or type(value.get("generation")) is not int
        or value["generation"] < 0
        or value.get("checkpoint_hash")
        != hash_json({k: v for k, v in value.items() if k != "checkpoint_hash"})
    ):
        raise ValueError("sealed SIM-only persistent actor/critic checkpoint required")
    consumed = value.get("consumed_numeric_batches")
    if (
        type(consumed) is not list
        or len(consumed) != value["generation"]
        or not all(_hash(item) for item in consumed)
        or len(set(consumed)) != len(consumed)
    ):
        raise ValueError("unique consumed numeric batches must survive checkpoint recovery")
    for key in ("actor", "anchor_actor"):
        policy.validate_model(value[key])
        if value[key]["schema"] != policy.TIME_AWARE_SCHEMA:
            raise ValueError("explicit time-aware v2 actor required")
    if value["actor"]["world_hash"] != value["anchor_actor"]["world_hash"]:
        raise ValueError("unchanged anchor world binding required")
    critic = value["critic"]
    if critic.get("architecture") != [127, 64, 1] or len(critic.get("layers", [])) != 2:
        raise ValueError("time-aware fixed critic architecture required")
    for layer, before, after in zip(critic["layers"], [127, 64], [64, 1], strict=True):
        _finite(layer["weight"], (after, before), bound=100)
        _finite(layer["bias"], (after,), bound=100)
    optimizer = value["optimizer"]
    step = optimizer["steps"]
    parameters = _parameters(value["actor"]["layers"] + critic["layers"])
    if (
        type(step) is not int
        or not 0 <= step <= 100000
        or step != value["actor"]["optimizer_steps"]
        or len(optimizer["slots"]) != len(parameters)
    ):
        raise ValueError("aligned persistent Adam step and full moment slots required")
    for slot, parameter in zip(optimizer["slots"], parameters, strict=True):
        average = _finite(slot["exp_avg"], parameter.shape, bound=1e6)
        square = _finite(slot["exp_avg_sq"], parameter.shape, bound=1e12)
        if np.any(square < 0):
            raise ValueError("nonnegative persistent Adam second moments required")
        if step == 0 and (np.any(average != 0) or np.any(square != 0)):
            raise ValueError("untrained optimizer must have zero initial moments")


def initialize_checkpoint(actor: dict[str, Any], *, anchor_features: Any) -> dict[str, Any]:
    policy.validate_model(actor)
    if actor["schema"] != policy.TIME_AWARE_SCHEMA or actor["optimizer_steps"] != 0:
        raise ValueError("new v2 actor required; never silently reset a learned optimizer")
    anchors = _anchors(anchor_features)
    rng = np.random.default_rng((actor["seed"] + 1) % 2**32)
    critic = dict(
        architecture=[127, 64, 1],
        layers=[
            dict(
                weight=(rng.normal(size=(64, 127)) / np.sqrt(127)).tolist(),
                bias=np.zeros(64).tolist(),
            ),
            dict(weight=np.zeros((1, 64)).tolist(), bias=[0.0]),
        ],
    )
    parameters = _parameters(actor["layers"] + critic["layers"])
    return _seal(
        dict(
            schema=SCHEMA,
            actor=copy.deepcopy(actor),
            anchor_actor=copy.deepcopy(actor),
            critic=critic,
            anchor_features_hash=_anchor_hash(anchors),
            generation=0,
            training_reward_contract_hash=reward_contract()["contract_hash"],
            consumed_numeric_batches=[],
            config=copy.deepcopy(CONFIG),
            optimizer=dict(
                steps=0,
                slots=[
                    dict(exp_avg=np.zeros_like(p).tolist(), exp_avg_sq=np.zeros_like(p).tolist())
                    for p in parameters
                ],
            ),
            source_hashes=dict(
                optimizer=hash_bytes(Path(__file__).read_bytes()),
                policy=hash_bytes(Path(policy.__file__).read_bytes()),
                reward=hash_bytes(Path(reward_module.__file__).read_bytes()),
            ),
            activation_ceiling="SIM_ONLY",
            hardware_authorized=False,
            promotion_authorized=False,
            runtime_execution_authorized=False,
            teacher_qualified=False,
            physics_qualified=False,
            fresh_holdout_open_authorized=False,
        )
    )


def _forward_numpy(layers: list[dict[str, Any]], features: np.ndarray) -> np.ndarray:
    value = features
    for index, layer in enumerate(layers):
        value = value @ np.asarray(layer["weight"]).T + np.asarray(layer["bias"])
        if index + 1 < len(layers):
            value = np.tanh(value)
    return value


def fit_update(
    checkpoint: dict[str, Any],
    arrays: dict[str, Any],
    *,
    batch_hash: str,
    anchor_features: Any,
    steps: int = 4,
) -> dict[str, Any]:
    """Optimize complete numeric episodes; physical provenance is caller-owned.

    All 3000 substeps, including every failed rollout, must be supplied. Task
    termination uses measured undiscounted terminal returns without bootstrap.
    No sample centering: equal failures still yield a nonzero REINFORCE signal.
    A KL-rejected joint update restores BOTH parameters and Adam moments.
    """
    validate_checkpoint(checkpoint)
    if type(arrays) is not dict:
        raise ValueError("explicit numeric episode dictionary required")
    anchors = _anchors(anchor_features)
    if _anchor_hash(anchors) != checkpoint["anchor_features_hash"]:
        raise ValueError("fixed retention anchor features changed across learning rounds")
    if type(steps) is not int or not 1 <= steps <= 8 or not _hash(batch_hash):
        raise ValueError("bounded update count and explicit source batch hash required")
    if checkpoint["optimizer"]["steps"] + steps > 100000:
        raise ValueError("persistent optimizer step budget would be exceeded")
    if arrays.get("behavior_model_hash") != checkpoint["actor"]["model_hash"]:
        raise ValueError("actual on-policy behavior actor binding required")
    if arrays.get("reward_contract_hash") != checkpoint["training_reward_contract_hash"]:
        raise ValueError("explicit fixed recovery training objective required")
    x = np.asarray(arrays["features"])
    if x.ndim != 2 or x.dtype != np.float64 or not 6000 <= len(x) <= 24000 or len(x) % 3000:
        raise ValueError("2..8 complete float64 3000-substep numeric episodes required")
    n = len(x)
    x = _finite(x, (n, 127), bound=10)
    if any(
        np.asarray(arrays[key]).dtype != np.float64
        for key in ("latent_action", "old_log_probability", "terminal_return")
    ):
        raise ValueError("float64 actual latent and terminal-return arrays required")
    z = _finite(arrays["latent_action"], (n, 29), bound=1e4)
    old_logp = _finite(arrays["old_log_probability"], (n,), bound=1e6)
    returns = _finite(arrays["terminal_return"], (n,), bound=1000)
    m = n // 3000
    group = np.asarray(arrays["trajectory_index"])
    clock = np.tile((3000 - np.arange(3000)) / 3000, m)
    if (
        group.dtype != np.int64
        or not np.array_equal(group, np.repeat(np.arange(m), 3000))
        or not np.array_equal(x[:, -1], clock)
        or np.asarray(arrays["terminated"]).dtype != np.bool_
        or np.asarray(arrays["truncated"]).dtype != np.bool_
        or np.asarray(arrays["terminated"]).shape != (m,)
        or np.asarray(arrays["truncated"]).shape != (m,)
        or not np.all(arrays["terminated"])
        or np.any(arrays["truncated"])
        or not np.all(returns.reshape(m, 3000) == returns.reshape(m, 3000)[:, :1])
    ):
        raise ValueError("complete ordered finite-task episodes and terminal returns required")
    old_mean = _forward_numpy(checkpoint["actor"]["layers"], x)
    density = np.sum(
        -0.5 * ((z - old_mean) / 0.05) ** 2 - np.log(0.05) - 0.5 * np.log(2 * np.pi), axis=1
    )
    if not np.allclose(old_logp, density, rtol=0, atol=1e-8):
        raise ValueError("actual old latent likelihood differs from the behavior actor")
    numeric_batch_hash = hash_json(
        dict(
            behavior_model_hash=arrays["behavior_model_hash"],
            reward_contract_hash=arrays["reward_contract_hash"],
            arrays={
                key: dict(
                    dtype=np.asarray(arrays[key]).dtype.str,
                    shape=list(np.asarray(arrays[key]).shape),
                    bytes_hash=hash_bytes(np.ascontiguousarray(arrays[key]).tobytes()),
                )
                for key in (
                    "features",
                    "latent_action",
                    "old_log_probability",
                    "terminal_return",
                    "trajectory_index",
                    "terminated",
                    "truncated",
                )
            },
        )
    )
    if numeric_batch_hash in checkpoint["consumed_numeric_batches"]:
        raise ValueError("numeric batch already consumed; do not silently repeat failed updates")

    import torch

    def load(layers: list[dict[str, Any]]) -> list[Any]:
        return [
            torch.nn.Parameter(torch.tensor(p, dtype=torch.float64, device="cpu"))
            for p in _parameters(layers)
        ]

    def forward(parameters: list[Any], values: Any) -> Any:
        for i in range(0, len(parameters), 2):
            values = torch.nn.functional.linear(values, parameters[i], parameters[i + 1])
            if i + 2 < len(parameters):
                values = torch.tanh(values)
        return values

    actor, critic = load(checkpoint["actor"]["layers"]), load(checkpoint["critic"]["layers"])
    parameters = actor + critic
    optimizer = torch.optim.Adam(
        parameters, lr=CONFIG["learning_rate"], betas=tuple(CONFIG["betas"]), eps=CONFIG["eps"]
    )
    for p, slot in zip(parameters, checkpoint["optimizer"]["slots"], strict=True):
        optimizer.state[p] = dict(
            step=torch.tensor(float(checkpoint["optimizer"]["steps"]), dtype=torch.float64),
            exp_avg=torch.tensor(slot["exp_avg"], dtype=torch.float64),
            exp_avg_sq=torch.tensor(slot["exp_avg_sq"], dtype=torch.float64),
        )
    obs, action, old_density, targets, anchor_obs = [
        torch.tensor(v, dtype=torch.float64) for v in (x, z, old_logp, returns / 10, anchors)
    ]
    old = torch.tensor(old_mean, dtype=torch.float64)
    anchor_mean = torch.tensor(
        _forward_numpy(checkpoint["anchor_actor"]["layers"], anchors), dtype=torch.float64
    )
    with torch.no_grad():
        advantage = (targets - forward(critic, obs)[:, 0]).clamp(-5, 5)

    def kl_statistics() -> dict[str, float]:
        rollout = ((forward(actor, obs) - old) ** 2 / (2 * 0.05**2)).sum(dim=1)
        retention = ((forward(actor, anchor_obs) - anchor_mean) ** 2 / (2 * 0.05**2)).sum(dim=1)
        return dict(
            rollout_mean=float(rollout.mean()),
            rollout_max=float(rollout.max()),
            anchor_mean=float(retention.mean()),
            anchor_max=float(retention.max()),
        )

    accepted, attempted, rejected = 0, 0, False
    actor_gradient_steps = 0
    for _ in range(steps):
        optimizer.zero_grad()
        mean = forward(actor, obs)
        logp = (-0.5 * ((action - mean) / 0.05) ** 2 - np.log(0.05) - 0.5 * np.log(2 * np.pi)).sum(
            dim=1
        )
        ratio = torch.exp(logp - old_density)
        surrogate = torch.minimum(ratio * advantage, ratio.clamp(0.8, 1.2) * advantage).mean()
        critic_loss = ((forward(critic, obs)[:, 0] - targets) ** 2).mean()
        anchor_kl = (
            ((forward(actor, anchor_obs) - anchor_mean) ** 2 / (2 * 0.05**2)).sum(dim=1).mean()
        )
        loss = -surrogate + 0.5 * critic_loss + 0.1 * anchor_kl
        if not torch.isfinite(loss):
            raise ValueError("nonfinite actor/critic objective")
        loss.backward()
        if any(p.grad is None or not torch.isfinite(p.grad).all() for p in parameters):
            raise ValueError("complete finite actor/critic gradients required")
        nonzero_actor_gradient = any(bool(torch.any(p.grad != 0)) for p in actor)
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        old_parameters = [p.detach().clone() for p in parameters]
        old_optimizer = copy.deepcopy(optimizer.state_dict())
        optimizer.step()
        attempted += 1
        with torch.no_grad():
            kl = kl_statistics()
            valid = (
                all(bool(torch.isfinite(p).all() and (p.abs() <= 100).all()) for p in parameters)
                and all(np.isfinite(v) for v in kl.values())
                and kl["rollout_mean"] <= CONFIG["maximum_rollout_mean_kl"]
                and kl["rollout_max"] <= CONFIG["maximum_rollout_state_kl"]
                and kl["anchor_mean"] <= CONFIG["maximum_anchor_mean_kl"]
                and kl["anchor_max"] <= CONFIG["maximum_anchor_state_kl"]
            )
            if not valid:
                for p, old_p in zip(parameters, old_parameters, strict=True):
                    p.copy_(old_p)
                optimizer.load_state_dict(old_optimizer)
                rejected = True
                break
        accepted += 1
        actor_gradient_steps += int(nonzero_actor_gradient)

    if any(
        int(optimizer.state[p]["step"].item()) != checkpoint["optimizer"]["steps"] + accepted
        for p in parameters
    ):
        raise ValueError("actual persistent Adam step counters differ from accepted updates")

    def export(parameters: list[Any]) -> list[dict[str, Any]]:
        return [
            dict(
                weight=parameters[i].detach().numpy().tolist(),
                bias=parameters[i + 1].detach().numpy().tolist(),
            )
            for i in range(0, len(parameters), 2)
        ]

    result = copy.deepcopy(checkpoint)
    result["actor"]["layers"] = export(actor)
    result["actor"]["optimizer_steps"] += accepted
    result["actor"]["model_hash"] = hash_json(
        {k: v for k, v in result["actor"].items() if k != "model_hash"}
    )
    result["critic"]["layers"] = export(critic)
    result["optimizer"] = dict(
        steps=checkpoint["optimizer"]["steps"] + accepted,
        slots=[
            dict(
                exp_avg=optimizer.state[p]["exp_avg"].numpy().tolist(),
                exp_avg_sq=optimizer.state[p]["exp_avg_sq"].numpy().tolist(),
            )
            for p in parameters
        ],
    )
    result["generation"] += 1
    result["consumed_numeric_batches"].append(numeric_batch_hash)
    with torch.no_grad():
        kl = kl_statistics()
    result["learning_receipt"] = dict(
        parent_checkpoint_hash=checkpoint["checkpoint_hash"],
        batch_hash_provided_by_caller=batch_hash,
        numeric_batch_hash=numeric_batch_hash,
        requested_optimizer_steps=steps,
        attempted_optimizer_steps=attempted,
        accepted_joint_optimizer_steps=accepted,
        nonzero_actor_gradient_steps=actor_gradient_steps,
        rejected_trial_rolled_back_with_adam_state=rejected,
        kl=kl,
        complete_numeric_episode_count=m,
        substep_sample_count=n,
        physical_provenance_verified_by_optimizer=False,
        physical_qualification_claimed=False,
        torch_version=torch.__version__,
        training_device="cpu",
        training_dtype="float64",
    )
    return _seal(result)
