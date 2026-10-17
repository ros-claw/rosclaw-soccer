"""Supervised SIM option selection, not a learned joint controller or online RL.

Only initial public ball x/y/vx enter inference. Complete success/failure
matrices are training labels. Callers authenticate all physical evidence and
freeze disjoint validation declarations before fitting. No teacher admission,
simulator callback, private exam, motor transport, or policy activation here.
"""

from __future__ import annotations

import math
import re
from typing import Any

import numpy as np

from rosclaw_soccer.rsi.initial_context_control_portfolio import ControlOutcome
from rosclaw_soccer.sim.contracts import hash_json

SCHEMA = "soccer.rsi.neural_contact_option_policy.v1"
WIDTHS = (3, 32, 32, 8)
_HASH = re.compile(r"sha256:[0-9a-f]{64}")


def _digest(value: Any) -> None:
    if type(value) is not str or _HASH.fullmatch(value) is None:
        raise ValueError("content-bound training/catalog/world identity required")


def features(context: tuple[float, float, float]) -> np.ndarray[Any, Any]:
    if (
        type(context) is not tuple
        or len(context) != 3
        or any(type(value) is not float or not math.isfinite(value) for value in context)
    ):
        raise ValueError("three finite initial public ball measurements required")
    x, y, vx = context
    if not ((2.2 <= x <= 2.4 or 2.6 <= x <= 2.8) and abs(y) <= 0.16 and 0.25 <= abs(vx) <= 0.7):
        raise ValueError("original public first-touch domain required; no clipping")
    return np.asarray(((x - 2.5) / 0.3, y / 0.16, vx / 0.7), dtype=np.float64)


def validate_model(model: dict[str, Any]) -> None:
    if (
        type(model) is not dict
        or model.get("schema") != SCHEMA
        or model.get("architecture") != list(WIDTHS)
        or model.get("activation_ceiling") != "SIM_ONLY"
        or model.get("learning_kind") != "supervised_initial_state_option_selection"
        or any(
            model.get(key) is not False
            for key in (
                "teacher_qualified",
                "runtime_execution_authorized",
                "promotion_authorized",
                "hardware_authorized",
                "fresh_holdout_open_authorized",
            )
        )
        or type(model.get("optimizer_steps")) is not int
        or not 1 <= model["optimizer_steps"] <= 2000
        or type(model.get("seed")) is not int
        or not 0 <= model["seed"] < 2**32
        or type(model.get("option_hashes")) is not list
        or len(model["option_hashes"]) != 8
        or any(type(value) is not str for value in model["option_hashes"])
        or len(set(model["option_hashes"])) != 8
        or type(model.get("layers")) is not list
        or len(model["layers"]) != 3
    ):
        raise ValueError("bounded supervised SIM option policy required")
    for value in [
        *model["option_hashes"],
        model.get("training_data_hash"),
        model.get("execution_binding_hash"),
    ]:
        _digest(value)
    for layer, before, after in zip(model["layers"], WIDTHS[:-1], WIDTHS[1:], strict=True):
        if type(layer) is not dict or set(layer) != {"weight", "bias"}:
            raise ValueError("complete fixed neural layer required")
        for key, shape in (("weight", (after, before)), ("bias", (after,))):
            value = np.asarray(layer[key], dtype=np.float64)
            if value.shape != shape or not np.isfinite(value).all() or np.max(np.abs(value)) > 100:
                raise ValueError("finite bounded complete neural parameters required")
    if model.get("model_hash") != hash_json(
        {key: value for key, value in model.items() if key != "model_hash"}
    ):
        raise ValueError("whole neural model seal differs")


def scores(model: dict[str, Any], context: tuple[float, float, float]) -> np.ndarray[Any, Any]:
    validate_model(model)
    value = features(context)
    for index, layer in enumerate(model["layers"]):
        value = np.asarray(layer["weight"], dtype=np.float64) @ value + np.asarray(
            layer["bias"], dtype=np.float64
        )
        if index < 2:
            value = np.tanh(value)
    if not np.isfinite(value).all():
        raise ValueError("finite option scores required")
    return value


def select_option(
    model: dict[str, Any],
    context: tuple[float, float, float],
    *,
    option_hashes: tuple[str, ...],
    execution_binding_hash: str,
) -> dict[str, Any]:
    value = scores(model, context)
    if (
        list(option_hashes) != model["option_hashes"]
        or execution_binding_hash != model["execution_binding_hash"]
    ):
        raise ValueError("actual option catalog and physical execution binding must match")
    index = int(np.argmax(value))
    result = dict(
        schema="soccer.rsi.neural_contact_option_selection.v1",
        model_hash=model["model_hash"],
        context=list(context),
        option_index=index,
        option_hash=option_hashes[index],
        scores=value.tolist(),
        evaluated_task_outcome_is_input=False,
        scores_are_calibrated_success_probabilities=False,
        success_or_safety_guaranteed=False,
        teacher_qualified=False,
        promotion_authorized=False,
        hardware_authorized=False,
    )
    return dict(result, selection_hash=hash_json(result))


def fit(
    contexts: tuple[tuple[float, float, float], ...],
    outcomes: tuple[tuple[ControlOutcome, ...], ...],
    *,
    sample_weights: tuple[float, ...],
    option_hashes: tuple[str, ...],
    execution_binding_hash: str,
    training_data_hash: str,
    seed: int,
    steps: int = 1000,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Fixed full-batch Adam, all labels retained; never checkpoint-select on exams."""
    if (
        type(contexts) is not tuple
        or not 4 <= len(contexts) <= 160
        or type(outcomes) is not tuple
        or len(outcomes) != len(contexts)
        or type(sample_weights) is not tuple
        or len(sample_weights) != len(contexts)
        or any(
            type(weight) is not float or not math.isfinite(weight) or not 0 < weight <= 4
            for weight in sample_weights
        )
        or type(option_hashes) is not tuple
        or len(option_hashes) != 8
        or any(type(value) is not str for value in option_hashes)
        or len(set(option_hashes)) != 8
        or type(seed) is not int
        or not 0 <= seed < 2**32
        or type(steps) is not int
        or not 1 <= steps <= 2000
    ):
        raise ValueError("bounded complete weighted public training matrix required")
    for value in (*option_hashes, execution_binding_hash, training_data_hash):
        _digest(value)
    x = np.stack([features(context) for context in contexts])
    if len(set(contexts)) != len(contexts):
        raise ValueError("repeated physical initial contexts cannot inflate training rows")
    for row in outcomes:
        if type(row) is not tuple or len(row) != 8:
            raise ValueError("all eight success and failure labels required per context")
        for outcome in row:
            if type(outcome) is not ControlOutcome:
                raise ValueError("typed measured task outcomes required")
            outcome.__post_init__()
    import torch

    # Training is CPU float64; no global seed mutation or serialized pickle.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)  # type: ignore[attr-defined]
        network = torch.nn.Sequential(
            torch.nn.Linear(3, 32, dtype=torch.float64, device="cpu"),
            torch.nn.Tanh(),
            torch.nn.Linear(32, 32, dtype=torch.float64, device="cpu"),
            torch.nn.Tanh(),
            torch.nn.Linear(32, 8, dtype=torch.float64, device="cpu"),
        )
        initial = hash_json([parameter.detach().tolist() for parameter in network.parameters()])
        optimizer = torch.optim.Adam(network.parameters(), lr=0.003)
        inputs = torch.from_numpy(x)
        targets = torch.tensor(
            [[float(item.task_pass) for item in row] for row in outcomes], dtype=torch.float64
        )
        weights = torch.tensor(sample_weights, dtype=torch.float64)
        feasible = targets.sum(dim=1) > 0
        losses = []
        for _ in range(steps):
            optimizer.zero_grad()
            logits = network(inputs)
            binary = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, targets, reduction="none"
            ).mean(dim=1)
            # Multi-label set loss: any actual passing option is acceptable.
            ranking = torch.zeros(len(contexts), dtype=torch.float64)
            if bool(feasible.any()):
                ranking[feasible] = torch.logsumexp(logits[feasible], dim=1) - torch.logsumexp(
                    logits[feasible].masked_fill(targets[feasible] == 0, -torch.inf), dim=1
                )
            loss = ((ranking + 0.25 * binary) * weights).sum() / weights.sum()
            if not bool(torch.isfinite(loss)):
                raise ValueError("nonfinite training loss; do not export candidate")
            loss.backward()  # type: ignore[no-untyped-call]
            optimizer.step()
            losses.append(float(loss.detach()))
        layers = [
            dict(weight=layer.weight.detach().tolist(), bias=layer.bias.detach().tolist())
            for layer in network
            if isinstance(layer, torch.nn.Linear)
        ]
        model = dict(
            schema=SCHEMA,
            architecture=list(WIDTHS),
            layers=layers,
            seed=seed,
            optimizer_steps=steps,
            option_hashes=list(option_hashes),
            execution_binding_hash=execution_binding_hash,
            training_data_hash=training_data_hash,
            activation_ceiling="SIM_ONLY",
            learning_kind="supervised_initial_state_option_selection",
            teacher_qualified=False,
            runtime_execution_authorized=False,
            promotion_authorized=False,
            hardware_authorized=False,
            fresh_holdout_open_authorized=False,
        )
        model["model_hash"] = hash_json(model)
        validate_model(model)
        expected = network(inputs).detach().numpy()
        restored = np.stack([scores(model, context) for context in contexts])
        if not np.allclose(expected, restored, rtol=0, atol=1e-11):
            raise ValueError("serialized independent NumPy inference differs from training network")
        final = hash_json([parameter.detach().tolist() for parameter in network.parameters()])
    report = dict(
        schema="soccer.rsi.neural_contact_option_fit.v1",
        model_hash=model["model_hash"],
        initial_parameter_hash=initial,
        final_parameter_hash=final,
        parameters_changed=initial != final,
        optimizer_steps=steps,
        training_contexts=len(contexts),
        feasible_training_contexts=int(feasible.sum()),
        selected_training_passes=sum(
            outcomes[i][int(np.argmax(row))].task_pass for i, row in enumerate(restored)
        ),
        first_loss=losses[0],
        last_loss=losses[-1],
        numpy_reload_maximum_error=float(np.max(np.abs(expected - restored))),
        resubstitution_only=True,
        new_physical_executions=0,
        football_improvement_proven=False,
        teacher_qualified=False,
        promotion_authorized=False,
        hardware_authorized=False,
    )
    return model, dict(report, report_hash=hash_json(report))
