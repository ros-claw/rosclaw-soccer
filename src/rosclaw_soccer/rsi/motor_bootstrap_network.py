"""Offline neural actor/critic warm start, not online RL or a runtime policy.

Optional torch is used only inside fitting. JSON numerical weights permit
inspection without pickle. Actor predicts bounded motor parameters from causal
proprioception; critic is an outcome regressor, not a qualified value function.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import numpy as np

from rosclaw_soccer.rsi.precontact_proprio_policy import FEATURE_NAMES
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json


def _array(values: Any, shape: tuple[int, ...]) -> np.ndarray[Any, Any]:
    result = np.asarray(values, dtype=np.float64)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError("finite aligned bootstrap network array required")
    return result


def _evaluate(network: dict[str, Any], values: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    dimensions = network["dimensions"]
    if dimensions not in ([13, 32, 37], [50, 32, 3]):
        raise ValueError("unsupported bootstrap architecture")
    values = _array(values, (dimensions[0],))
    mean = _array(network["mean"], (dimensions[0],))
    scale = _array(network["scale"], (dimensions[0],))
    if np.any(scale <= 0) or len(network["layers"]) != 2:
        raise ValueError("positive scale and two numerical layers required")
    hidden = (values - mean) / scale
    for index, layer in enumerate(network["layers"]):
        weight = _array(layer["weight"], (dimensions[index + 1], dimensions[index]))
        bias = _array(layer["bias"], (dimensions[index + 1],))
        hidden = weight @ hidden + bias
        if index == 0:
            hidden = np.tanh(hidden)
    if not np.isfinite(hidden).all():
        raise ValueError("nonfinite bootstrap prediction")
    return cast(np.ndarray[Any, Any], hidden)


def validate_model(model: dict[str, Any]) -> None:
    bank_hash = model.get("bank_hash")
    if (
        model.get("schema") != "soccer.rsi.offline_motor_actor_critic_bootstrap.v1"
        or model.get("activation_ceiling") != "SIM_ONLY"
        or model.get("promotion_authorized") is not False
        or model.get("runtime_execution_authorized") is not False
        or model.get("fresh_holdout_open_authorized") is not False
        or model.get("learning_kind") != "offline_imitation_and_outcome_regression"
        or not isinstance(bank_hash, str)
        or not bank_hash.startswith("sha256:")
        or len(bank_hash) != 71
        or any(character not in "0123456789abcdef" for character in bank_hash[7:])
        or model.get("observation_names") != list(FEATURE_NAMES)
        or model.get("source_hash") != hash_bytes(Path(__file__).read_bytes())
        or model.get("model_hash")
        != hash_json({k: v for k, v in model.items() if k != "model_hash"})
        or model.get("actor", {}).get("dimensions") != [13, 32, 37]
        or model.get("critic", {}).get("dimensions") != [50, 32, 3]
    ):
        raise ValueError("unsealed offline-only neural bootstrap")
    _evaluate(model["actor"], np.zeros(13))
    _evaluate(model["critic"], np.zeros(50))


def actor_parameters(model: dict[str, Any], features: tuple[float, ...]) -> np.ndarray[Any, Any]:
    """Diagnostic inference only: artifact does NOT authorize applying targets."""
    validate_model(model)
    normalized = np.clip(_evaluate(model["actor"], np.asarray(features)), -1.0, 1.0)
    return np.concatenate((normalized[:36] * 0.16, [normalized[36] * 0.3 - 0.05]))


def fit_bootstrap(bank: dict[str, Any], *, epochs: int = 2500) -> dict[str, Any]:
    quarantine_path = (
        Path(__file__).resolve().parents[3]
        / "docs/rsi/protocols/neural-motor-fresh-quarantine-v307.json"
    )
    quarantine = json.loads(quarantine_path.read_text(encoding="utf-8"))
    if (
        quarantine.get("partition") != "FRESH_QUARANTINED_NOT_OPENED"
        or quarantine.get("data_generated") is not False
        or quarantine.get("labels_observed") is not False
        or quarantine.get("automatic_open_authorized") is not False
    ):
        raise ValueError("sealed unopened fresh protocol required before neural fitting")
    if (
        bank.get("schema") != "soccer.rsi.causal_motor_actor_critic_bootstrap_bank.v1"
        or bank.get("report_hash")
        != hash_json({k: v for k, v in bank.items() if k != "report_hash"})
        or bank.get("partition") != "TRAIN_CONSUMED"
        or bank.get("sample_count") != 384
        or len(bank.get("critic_samples", [])) != 384
        or bank.get("distinct_consumed_course_count") != 12
        or bank.get("observation_names") != list(FEATURE_NAMES)
        or bank.get("runtime_selection_authorized") is not False
        or bank.get("promotion_authorized") is not False
        or bank.get("fresh_holdout_open_authorized") is not False
        or type(epochs) is not int
        or not 1 <= epochs <= 10000
    ):
        raise ValueError("sealed completed consumed bank and bounded epoch count required")
    samples = bank["critic_samples"]
    teachers = bank["actor_teacher_samples"]
    if len(teachers) < 2 or len(teachers) != bank["successful_teacher_course_count"]:
        raise ValueError("at least two observed successful teacher courses required")
    contexts = {
        (r["audit_metadata"]["course"][0], r["audit_metadata"]["course"][1]) for r in samples
    }
    if len(contexts) != 12:
        raise ValueError("12 distinct physical contexts required, not duplicated frames")
    x_actor = _array([r["observation"] for r in teachers], (len(teachers), 13))
    y_actor = _array([r["teacher_motor_parameters"] for r in teachers], (len(teachers), 37))
    x_critic = _array([r["observation"] + r["motor_parameters"] for r in samples], (384, 50))
    for parameters in (y_actor, x_critic[:, 13:]):
        if (
            np.any(np.abs(parameters[:, :36]) > 0.16)
            or np.any(parameters[:, 36] < -0.35)
            or np.any(parameters[:, 36] > 0.25)
        ):
            raise ValueError("teacher/proposal outside motor envelope")
    labels = []
    observed_contexts: dict[tuple[int, int], tuple[float, ...]] = {}
    successful_actions = set()
    course_counts: dict[tuple[int, int], int] = {}
    for sample in samples:
        row = sample["learning_labels"]
        if any(type(row[key]) is not bool for key in ("high_quality", "clean_foot_only")):
            raise ValueError("physical outcome booleans required")
        course = tuple(sample["audit_metadata"]["course"])
        observation = tuple(sample["observation"])
        if course in observed_contexts and observed_contexts[course] != observation:
            raise ValueError("unfrozen context across physical candidate branches")
        observed_contexts[course] = observation
        course_counts[course] = course_counts.get(course, 0) + 1
        if row["high_quality"]:
            successful_actions.add((observation, tuple(sample["motor_parameters"])))
        labels.append([row["reward"], float(row["high_quality"]), float(row["clean_foot_only"])])
    if any(count != 32 for count in course_counts.values()) or any(
        (tuple(teacher["observation"]), tuple(teacher["teacher_motor_parameters"]))
        not in successful_actions
        for teacher in teachers
    ):
        raise ValueError(
            "complete matched branches and genuinely successful actor teachers required"
        )
    y_critic = _array(labels, (384, 3))
    _, unique_indices, inverse = np.unique(x_critic, axis=0, return_index=True, return_inverse=True)
    for index, group in enumerate(inverse):
        if not np.array_equal(y_critic[index], y_critic[unique_indices[group]]):
            raise ValueError("identical condition/action has inconsistent physical labels")
    # Identical replay anchors are retained as evidence, not counted repeatedly
    # as extra learning diversity. This remains 12 contexts, not 384 situations.
    x_critic, y_critic = x_critic[unique_indices], y_critic[unique_indices]
    # Imports are optional for the runtime package. No driver, simulator or SDK.
    import torch

    torch.manual_seed(20261001307)
    torch.set_num_threads(1)
    networks = []
    diagnostics = {
        "unique_critic_condition_action_count": float(len(unique_indices)),
        "raw_physical_feedback_count": 384.0,
    }
    for name, x, y, dimensions in (
        ("actor", x_actor, y_actor.copy(), [13, 32, 37]),
        ("critic", x_critic, y_critic.copy(), [50, 32, 3]),
    ):
        mean = x.mean(axis=0)
        scale = np.maximum(x.std(axis=0), 1e-4)
        target_mean, target_scale = 0.0, 1.0
        if name == "actor":
            y[:, :36] /= 0.16
            y[:, 36] = (y[:, 36] + 0.05) / 0.3
        else:
            target_mean = float(y[:, 0].mean())
            target_scale = max(float(y[:, 0].std()), 1e-4)
            y[:, 0] = (y[:, 0] - target_mean) / target_scale
        tensor_x = torch.from_numpy((x - mean) / scale)
        tensor_y = torch.from_numpy(y)
        first_layer = torch.nn.Linear(dimensions[0], 32, device="cpu", dtype=torch.float64)
        last_layer = torch.nn.Linear(32, dimensions[-1], device="cpu", dtype=torch.float64)
        network = torch.nn.Sequential(first_layer, torch.nn.Tanh(), last_layer)
        optimizer = torch.optim.Adam(network.parameters(), lr=0.003)
        for _ in range(epochs):
            prediction = network(tensor_x)
            if name == "actor":
                loss = torch.nn.functional.mse_loss(prediction, tensor_y)
            else:
                loss = torch.nn.functional.mse_loss(
                    prediction[:, 0], tensor_y[:, 0]
                ) + torch.nn.functional.binary_cross_entropy_with_logits(
                    prediction[:, 1:], tensor_y[:, 1:]
                )
            if not bool(torch.isfinite(loss)):
                raise ValueError("nonfinite bootstrap optimization")
            optimizer.zero_grad()
            torch.autograd.backward(loss)
            optimizer.step()
        exported = {
            "dimensions": dimensions,
            "mean": mean.tolist(),
            "scale": scale.tolist(),
            "layers": [
                {
                    "weight": layer.weight.detach().numpy().tolist(),
                    "bias": layer.bias.detach().numpy().tolist(),
                }
                for layer in (first_layer, last_layer)
            ],
            "reward_target_mean": target_mean,
            "reward_target_scale": target_scale,
        }
        networks.append(exported)
        diagnostics[f"{name}_training_loss"] = float(loss.detach())
        # Numerical export parity must hold; serialized inference is the artifact.
        for item, prediction in zip(x, network(tensor_x).detach().numpy(), strict=True):
            if not np.allclose(_evaluate(exported, item), prediction, atol=1e-10, rtol=1e-10):
                raise ValueError("JSON network differs from fitted numerical model")
    model = {
        "schema": "soccer.rsi.offline_motor_actor_critic_bootstrap.v1",
        "activation_ceiling": "SIM_ONLY",
        "learning_kind": "offline_imitation_and_outcome_regression",
        "observation_names": list(FEATURE_NAMES),
        "bank_hash": bank["report_hash"],
        "fresh_quarantine_protocol_hash": hash_json(quarantine),
        "source_hash": hash_bytes(Path(__file__).read_bytes()),
        "epochs": epochs,
        "actor": networks[0],
        "critic": networks[1],
        "diagnostics": diagnostics,
        "evaluation_partition": "TRAIN_CONSUMED_ONLY",
        "runtime_execution_authorized": False,
        "promotion_authorized": False,
        "fresh_holdout_open_authorized": False,
        "not_claimed": [
            "online RL",
            "qualified critic",
            "physical actor improvement",
            "neural torque policy",
            "fresh generalization",
        ],
    }
    model["model_hash"] = hash_json(model)
    validate_model(model)
    return model
