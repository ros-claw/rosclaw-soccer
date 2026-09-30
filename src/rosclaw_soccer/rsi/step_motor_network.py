"""Offline per-frame neural motor warm start, never a motion authorization.

Supervised successful-action distillation plus terminal-outcome regression.
Not online PPO, not a qualified value function, and not a torque policy. The
physics caller must evaluate closed-loop distribution shift and retention.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import numpy as np

from rosclaw_soccer.rsi import step_motor_features as feature_module
from rosclaw_soccer.rsi.contact_motor_primitive import CAP_RAD, JOINT_NAMES, SLEW_RAD
from rosclaw_soccer.rsi.step_motor_features import FEATURE_NAMES
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json

SCHEMA = "soccer.rsi.offline_step_motor_actor_outcome_critic.v1"


def validate_model(model: dict[str, Any]) -> None:
    if (
        model.get("schema") != SCHEMA
        or model.get("activation_ceiling") != "SIM_ONLY"
        or model.get("learning_kind") != "offline_success_distillation_and_terminal_regression"
        or any(
            model.get(k) is not False
            for k in (
                "physics_qualified",
                "online_rl_qualified",
                "promotion_authorized",
                "runtime_execution_authorized",
                "hardware_authorized",
                "fresh_holdout_open_authorized",
            )
        )
        or model.get("feature_names") != list(FEATURE_NAMES)
        or model.get("action_joint_names") != list(JOINT_NAMES)
        or model.get("source_hash") != hash_bytes(Path(__file__).read_bytes())
        or model.get("feature_source_hash")
        != hash_bytes(Path(feature_module.__file__).read_bytes())
        or model.get("model_hash")
        != hash_json({k: v for k, v in model.items() if k != "model_hash"})
        or model.get("cap_rad") != CAP_RAD
        or model.get("slew_rad_per_frame") != SLEW_RAD
    ):
        raise ValueError("unsealed offline-only per-frame motor model")
    for key in ("mean", "scale"):
        values = np.asarray(model[key], dtype=np.float64)
        if (
            values.shape != (134,)
            or not np.isfinite(values).all()
            or (key == "scale" and np.any(values < 0.05))
        ):
            raise ValueError("finite training-only feature normalization required")
    for name, dimensions in (("actor", (134, 128, 128, 12)), ("critic", (134, 128, 1))):
        network = model[name]
        if (
            network["dimensions"] != list(dimensions)
            or len(network["layers"]) != len(dimensions) - 1
        ):
            raise ValueError("explicit fixed neural architecture required")
        for index, layer in enumerate(network["layers"]):
            w, b = np.asarray(layer["weight"]), np.asarray(layer["bias"])
            if (
                w.shape != (dimensions[index + 1], dimensions[index])
                or b.shape != (dimensions[index + 1],)
                or not np.isfinite(w).all()
                or not np.isfinite(b).all()
            ):
                raise ValueError("finite aligned neural weights required")


def _evaluate(model: dict[str, Any], features: Any, name: str) -> np.ndarray[Any, Any]:
    validate_model(model)
    values = np.asarray(features, dtype=np.float64)
    if values.shape != (134,) or not np.isfinite(values).all():
        raise ValueError("finite aligned causal motor features required")
    values = np.clip((values - np.asarray(model["mean"])) / np.asarray(model["scale"]), -8.0, 8.0)
    layers = model[name]["layers"]
    for index, layer in enumerate(layers):
        values = np.asarray(layer["weight"]) @ values + np.asarray(layer["bias"])
        if index < len(layers) - 1 or name == "actor":
            values = np.tanh(values)
    return cast(np.ndarray[Any, Any], values)


def predict_delta(
    model: dict[str, Any],
    features: Any,
    *,
    baseline: Any,
    limits: Any,
    previous: Any,
    frame: int,
) -> np.ndarray[Any, Any]:
    base, bounds, prior = [np.asarray(v, dtype=np.float64) for v in (baseline, limits, previous)]
    if (
        base.shape != (12,)
        or bounds.shape != (12, 2)
        or prior.shape != (12,)
        or not all(np.isfinite(v).all() for v in (base, bounds, prior))
        or np.any(bounds[:, 0] >= bounds[:, 1])
        or np.max(np.abs(prior)) > CAP_RAD + 1e-5
        or type(frame) is not int
        or not 0 <= frame < 3000
    ):
        raise ValueError("bounded measured per-frame motor boundary required")
    desired = _evaluate(model, features, "actor") * CAP_RAD
    if frame < 30:
        if np.any(prior != 0):
            raise ValueError("residual cannot predate causal decision boundary")
        return np.zeros(12)
    proposed = prior + np.clip(desired - prior, -SLEW_RAD, SLEW_RAD)
    lower = np.maximum(np.minimum(0.0, bounds[:, 0] - base), -CAP_RAD)
    upper = np.minimum(np.maximum(0.0, bounds[:, 1] - base), CAP_RAD)
    return cast(np.ndarray[Any, Any], np.clip(proposed, lower, upper))


def fit_model(manifest: dict[str, Any], arrays: Any, *, epochs: int = 200) -> dict[str, Any]:
    import torch

    if (
        manifest.get("schema") != "soccer.rsi.causal_step_motor_teacher_bank.v1"
        or manifest.get("partition") != "TRAIN_CONSUMED"
        or manifest.get("feature_names") != list(FEATURE_NAMES)
        or manifest.get("report_hash")
        != hash_json({k: v for k, v in manifest.items() if k != "report_hash"})
        or type(epochs) is not int
        or not 1 <= epochs <= 1000
    ):
        raise ValueError("sealed consumed physical teacher bank and bounded fitting required")
    x = np.asarray(arrays["observation"], dtype=np.float64)
    action = np.asarray(arrays["applied_delta_rad"], dtype=np.float64)
    outcome = np.asarray(arrays["terminal_return_label"], dtype=np.float64)
    positive = np.asarray(arrays["actor_supervision_mask"])
    groups = np.asarray(arrays["trajectory_index"])
    n = manifest["sample_count"]
    if (
        x.shape != (n, 134)
        or action.shape != (n, 12)
        or outcome.shape != (n,)
        or positive.shape != (n,)
        or positive.dtype != np.dtype(bool)
        or groups.shape != (n,)
        or not all(np.isfinite(a).all() for a in (x, action, outcome, groups))
        or np.max(np.abs(action)) > CAP_RAD + 2e-5
    ):
        raise ValueError("bounded aligned physical supervision required")
    trajectories = manifest["trajectories"]
    seed_by_group = {r["index"]: r["seed"] for r in trajectories}
    if set(groups.tolist()) != set(seed_by_group):
        raise ValueError("complete physical trajectory grouping required")
    seeds = sorted(set(seed_by_group.values()))
    if len(seeds) < 10:
        raise ValueError("sufficient independent consumed seed groups required")
    rng = np.random.default_rng(20261001314)
    validation_seeds = set(rng.choice(seeds, size=max(2, len(seeds) // 5), replace=False).tolist())
    train = np.asarray([seed_by_group[int(g)] not in validation_seeds for g in groups])
    actor_train = np.flatnonzero(train & positive)
    actor_val = np.flatnonzero(~train & positive)
    if min(len(actor_train), len(actor_val)) < 100:
        raise ValueError("nonempty grouped train and validation successes required")
    mean = x[train].mean(axis=0)
    scale = np.maximum(x[train].std(axis=0), 0.05)
    values = torch.tensor(np.clip((x - mean) / scale, -8.0, 8.0), dtype=torch.float32)
    targets = torch.tensor(np.clip(action / CAP_RAD, -1.0, 1.0), dtype=torch.float32)
    return_scale = max(1.0, float(np.std(outcome[train])))
    return_mean = float(np.mean(outcome[train]))
    return_targets = torch.tensor((outcome - return_mean) / return_scale, dtype=torch.float32)[
        :, None
    ]
    torch.set_num_threads(4)
    torch.manual_seed(20261001314)
    torch.use_deterministic_algorithms(True)
    actor = torch.nn.Sequential(
        torch.nn.Linear(134, 128),
        torch.nn.Tanh(),
        torch.nn.Linear(128, 128),
        torch.nn.Tanh(),
        torch.nn.Linear(128, 12),
        torch.nn.Tanh(),
    )
    critic = torch.nn.Sequential(
        torch.nn.Linear(134, 128), torch.nn.Tanh(), torch.nn.Linear(128, 1)
    )
    optimizer = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=3e-4)
    all_train = np.flatnonzero(train)
    for _ in range(epochs):
        actor_ids = rng.permutation(actor_train)
        critic_ids = rng.permutation(all_train)
        for offset in range(0, len(actor_ids), 1024):
            ids = actor_ids[offset : offset + 1024]
            value_ids = critic_ids[offset : offset + 1024]
            if not len(value_ids):
                value_ids = critic_ids[:1024]
            # Active corrections receive higher weight; zero phases remain taught.
            weight = 1 + 4 * (targets[ids].abs().amax(dim=1, keepdim=True) > 0.01)
            loss = (weight * (actor(values[ids]) - targets[ids]).square()).mean()
            loss = (
                loss + 0.1 * (critic(values[value_ids]) - return_targets[value_ids]).square().mean()
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(actor.parameters()) + list(critic.parameters()), 1.0
            )
            optimizer.step()
    with torch.no_grad():
        val_error = (actor(values[actor_val]) - targets[actor_val]).abs() * CAP_RAD
        active = targets[actor_val].abs().amax(dim=1) > 0.01
        metrics = dict(
            validation_mae_rad=float(val_error.mean()),
            active_validation_mae_rad=float(val_error[active].mean())
            if bool(active.any())
            else None,
            validation_seed_clusters=sorted(validation_seeds),
            training_seed_clusters=sorted(set(seeds) - validation_seeds),
            validation_supervised_frames=len(actor_val),
            training_supervised_frames=len(actor_train),
            interpretation="grouped offline teacher-forced error, not closed-loop success",
        )

    def export(network: Any, dimensions: list[int]) -> dict[str, Any]:
        return dict(
            dimensions=dimensions,
            layers=[
                dict(
                    weight=m.weight.detach().numpy().tolist(), bias=m.bias.detach().numpy().tolist()
                )
                for m in network
                if isinstance(m, torch.nn.Linear)
            ],
        )

    result = dict(
        schema=SCHEMA,
        activation_ceiling="SIM_ONLY",
        learning_kind="offline_success_distillation_and_terminal_regression",
        feature_names=list(FEATURE_NAMES),
        action_joint_names=list(JOINT_NAMES),
        actor=export(actor, [134, 128, 128, 12]),
        critic=export(critic, [134, 128, 1]),
        mean=mean.tolist(),
        scale=scale.tolist(),
        critic_return_mean=return_mean,
        critic_return_scale=return_scale,
        training_bank_hash=manifest["report_hash"],
        epochs=epochs,
        seed=20261001314,
        torch_version=torch.__version__,
        numpy_version=np.__version__,
        training_device="cpu",
        metrics=metrics,
        cap_rad=CAP_RAD,
        slew_rad_per_frame=SLEW_RAD,
        source_hash=hash_bytes(Path(__file__).read_bytes()),
        feature_source_hash=hash_bytes(Path(feature_module.__file__).read_bytes()),
        physics_qualified=False,
        online_rl_qualified=False,
        runtime_execution_authorized=False,
        promotion_authorized=False,
        hardware_authorized=False,
        fresh_holdout_open_authorized=False,
    )
    result["model_hash"] = hash_json(result)
    validate_model(result)
    return result
