"""Multistate bounded CEM proposals for a physics teacher, never admission.

Every candidate must be evaluated on every declared public initial state.
Incomplete evaluations and safety regressions cannot become CEM elites.
This searches SIM control parameters, not neural weights or private exams.
"""

import math
from dataclasses import asdict
from typing import Any

import numpy as np

from rosclaw_soccer.rsi.bounded_first_touch_control import BOUNDS, BoundedFirstTouchControl


def _rng(seed: int, count: int) -> np.random.Generator:
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("explicit uint32 search seed required")
    if type(count) is not int or not 2 <= count <= 32:
        raise ValueError("bounded population of two to32 required")
    return np.random.default_rng(seed)


def initial_population(*, seed: int, count: int) -> tuple[BoundedFirstTouchControl, ...]:
    rng = _rng(seed, count)
    result = [BoundedFirstTouchControl()]
    for _ in range(count - 1):
        result.append(
            BoundedFirstTouchControl(
                **{name: float(rng.uniform(lo, hi)) for name, (lo, hi) in BOUNDS.items()}
            )
        )
    return tuple(result)


def select_multistate_elites(
    population: tuple[BoundedFirstTouchControl, ...],
    *,
    state_ids: tuple[int, ...],
    records: list[dict[str, Any]],
    elite_count: int,
) -> tuple[int, ...]:
    if (
        type(population) is not tuple
        or not 2 <= len(population) <= 32
        or any(type(p) is not BoundedFirstTouchControl for p in population)
        or type(state_ids) is not tuple
        or len(state_ids) < 4
        or any(type(s) is not int or s < 0 for s in state_ids)
        or len(set(state_ids)) != len(state_ids)
        or type(elite_count) is not int
        or not 1 <= elite_count <= len(population)
    ):
        raise ValueError("complete bounded multistate search declaration required")
    expected = {(i, s) for i in range(len(population)) for s in state_ids}
    outcomes: dict[tuple[int, int], dict[str, Any]] = {}
    for row in records:
        i, s = row.get("candidate"), row.get("state")
        if type(i) is not int or type(s) is not int or (i, s) not in expected or (i, s) in outcomes:
            raise ValueError("exact unique candidate/state evaluations required")
        outcome = row.get("outcome")
        if (
            type(outcome) is not dict
            or type(outcome.get("safe")) is not bool
            or type(outcome.get("task_pass")) is not bool
            or type(outcome.get("training_reward")) is not float
            or not math.isfinite(outcome["training_reward"])
            or (outcome["task_pass"] and not outcome["safe"])
        ):
            raise ValueError("primitive finite complete physical outcomes required")
        outcomes[(i, s)] = outcome
    if set(outcomes) != expected:
        raise ValueError("no incomplete candidate or missing failure admission")
    ranked = []
    for i in range(len(population)):
        rows = [outcomes[(i, s)] for s in state_ids]
        if not all(row["safe"] for row in rows):
            continue
        reward = [row["training_reward"] for row in rows]
        score = (sum(row["task_pass"] for row in rows), min(reward), sum(reward) / len(reward))
        ranked.append((score, -i, i))
    ranked.sort(reverse=True)
    return tuple(row[2] for row in ranked[:elite_count])


def refine_population(
    population: tuple[BoundedFirstTouchControl, ...],
    elite_indices: tuple[int, ...],
    *,
    seed: int,
    count: int,
) -> tuple[BoundedFirstTouchControl, ...]:
    """Keep the best safe evaluated candidate; bound Gaussian exploration.

    No safe elite means stop, not silently retry with unsafe candidates. Caller
    must re-evaluate the new generation on the entire fixed state declaration.
    """
    rng = _rng(seed, count)
    if (
        type(population) is not tuple
        or not population
        or any(type(p) is not BoundedFirstTouchControl for p in population)
        or type(elite_indices) is not tuple
        or not elite_indices
        or len(set(elite_indices)) != len(elite_indices)
        or any(type(i) is not int or not 0 <= i < len(population) for i in elite_indices)
    ):
        raise ValueError("explicit safe evaluated elites required; otherwise stop search")
    names = list(BOUNDS)
    lower = np.asarray([BOUNDS[n][0] for n in names])
    span = np.asarray([BOUNDS[n][1] - BOUNDS[n][0] for n in names])
    values = np.asarray([[asdict(population[i])[n] for n in names] for i in elite_indices])
    normalized = (values - lower) / span
    mean, std = normalized.mean(axis=0), np.maximum(normalized.std(axis=0), 0.05)
    result = [population[elite_indices[0]]]
    # Rejection sampling, not boundary accumulation by clipping parameters.
    for _ in range(count - 1):
        for _attempt in range(10000):
            draw = rng.normal(mean, std)
            if np.all((draw >= 0.0) & (draw <= 1.0)):
                parameters = lower + span * draw
                result.append(
                    BoundedFirstTouchControl(
                        **{n: float(v) for n, v in zip(names, parameters, strict=True)}
                    )
                )
                break
        else:
            raise ValueError("bounded CEM rejection budget exhausted; preserve failed round")
    return tuple(result)
