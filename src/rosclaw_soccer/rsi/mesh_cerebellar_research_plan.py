"""Bounded V2 torque-learning jobs and zero-head retention inputs.

Public development only. No simulator, optimizer, private exam or promotion.
The original option catalog is a fixed prior, not a new-world teacher claim.
"""

from typing import Any

import numpy as np

from rosclaw_soccer.rsi.cerebellar_torque_policy import (
    CAPS,
    TIME_AWARE_SCHEMA,
    features,
    validate_model,
)
from rosclaw_soccer.rsi.public_physics_option_planning import PublicPlanningContext
from rosclaw_soccer.sim.contracts import hash_json

SEEDS = (2026101801, 2026101802, 2026101803)
OPTIONS = (3, 7, 0, 0, 2, 0, 1, 2)
RETENTION_GROUPS = (4, 8, 32, 56)


def build_jobs(
    contexts: tuple[PublicPlanningContext, ...], actors: dict[int, dict[str, Any]]
) -> list[dict[str, Any]]:
    if type(contexts) is not tuple or len(contexts) != 8 or tuple(actors) != SEEDS:
        raise ValueError("eight declared public contexts and all three initial seeds required")
    for seed, model in actors.items():
        validate_model(model)
        if (
            model["schema"] != TIME_AWARE_SCHEMA
            or model["seed"] != seed
            or model["optimizer_steps"] != 0
            or np.any(np.asarray(model["layers"][-1]["weight"]) != 0)
            or np.any(np.asarray(model["layers"][-1]["bias"]) != 0)
        ):
            raise ValueError("fresh seed-bound zero-step actors required")
    if len({model["world_hash"] for model in actors.values()}) != 1:
        raise ValueError("one exact new world across seeds required")
    for index, context in enumerate(contexts):
        if type(context) is not PublicPlanningContext or context.context_index != index:
            raise ValueError("ordered complete typed public context bank required")
        context.__post_init__()
    jobs: list[dict[str, Any]] = []

    def add(kind: str, seed: int, *, index: int | None = None, group: int | None = None) -> None:
        context = None if index is None else contexts[index]
        pair = f"raw-{group}" if context is None else f"context-{index}"
        sampling_seed = (
            int(
                hash_json(
                    dict(
                        domain="mesh_v2_cerebellar_collection_v1",
                        actor_seed=seed,
                        context_hash=context.context_hash,
                    )
                )[7:15],
                16,
            )
            if kind == "collection" and context is not None
            else None
        )
        jobs.append(
            dict(
                job=f"{kind}-{seed}-{pair}",
                kind=kind,
                actor_seed=seed,
                context_index=index,
                context_hash=None if context is None else context.context_hash,
                group=group if context is None else context.source_group,
                option_index=0 if index is None else OPTIONS[index],
                pair_id=pair,
                model_source="initial_checkpoint"
                if kind in ("zero_control", "collection")
                else "trained_checkpoint",
                model_hash=actors[seed]["model_hash"]
                if kind in ("zero_control", "collection")
                else None,
                sampling_seed=sampling_seed,
            )
        )

    for group in RETENTION_GROUPS:
        add("zero_control", SEEDS[0], group=group)
    for index in (1, 3, 5, 7):
        add("zero_control", SEEDS[0], index=index)
    for seed in SEEDS:
        for index in (0, 2, 4, 6):
            add("collection", seed, index=index)
    for seed in SEEDS:
        for index in (1, 3, 5, 7):
            add("evaluation", seed, index=index)
        for group in RETENTION_GROUPS:
            add("retention", seed, group=group)
    if (
        len(jobs) != 44
        or len({j["job"] for j in jobs}) != 44
        or len({j["sampling_seed"] for j in jobs if j["kind"] == "collection"}) != 12
    ):
        raise ValueError("44 unique jobs with 12 independent exploration streams required")
    return jobs


def zero_head_anchor_features(traces: list[dict[str, Any]]) -> np.ndarray:
    """Frame-start inputs reconstructed from authenticated passing None traces.

    Callers authenticate full receipts, task labels and original None-layer intent.
    These are 50Hz recorded states, not 500Hz trajectories or learned samples.
    """
    if type(traces) is not list or not 2 <= len(traces) <= 4:
        raise ValueError("at least two independently passing original public traces required")
    rows = []
    for trace in traces:
        if "applied_cerebellar_torque_offset_nm" in trace:
            raise ValueError("original None-layer references required, not assumed zero offsets")
        for key, shape in (
            ("canonical_qpos", (300, 1, 43)),
            ("canonical_qvel", (300, 1, 41)),
            ("joint_target_rad", (300, 1, 29)),
        ):
            value = np.asarray(trace.get(key))
            if value.shape != shape or value.dtype != np.float64 or not np.isfinite(value).all():
                raise ValueError("complete finite original frame-start states and targets required")
        for frame in range(300):
            rows.append(
                features(
                    trace["canonical_qpos"][frame, 0],
                    trace["canonical_qvel"][frame, 0],
                    trace["joint_target_rad"][frame, 0],
                    np.zeros(29),
                    substep_index=frame * 10,
                )
            )
    return np.asarray(rows, dtype=np.float64)


def same_input_torque_drift(
    before: dict[str, Any], after: dict[str, Any], inputs: np.ndarray
) -> dict[str, Any]:
    """Actual deterministic residual drift on fixed inputs, not a physics claim."""
    for actor in (before, after):
        validate_model(actor)
    if (
        before["world_hash"] != after["world_hash"]
        or before["schema"] != TIME_AWARE_SCHEMA
        or after["schema"] != TIME_AWARE_SCHEMA
        or inputs.ndim != 2
        or inputs.shape[1] != 127
        or not len(inputs)
        or inputs.dtype != np.float64
        or not np.isfinite(inputs).all()
    ):
        raise ValueError("same V2 world and complete finite fixed inputs required")

    def torque(actor: dict[str, Any]) -> np.ndarray:
        value = inputs
        for layer in actor["layers"]:
            value = np.tanh(value @ np.asarray(layer["weight"]).T + layer["bias"])
        return value * CAPS

    drift = torque(after) - torque(before)
    norms = np.linalg.norm(drift, axis=1)
    return dict(
        input_samples=len(inputs),
        before_model_hash=before["model_hash"],
        after_model_hash=after["model_hash"],
        mean_vector_drift_nm=float(np.mean(norms)),
        maximum_vector_drift_nm=float(np.max(norms)),
        rms_joint_drift_nm=float(np.sqrt(np.mean(drift**2))),
        maximum_absolute_joint_drift_nm=float(np.max(np.abs(drift))),
        deterministic_same_input_only=True,
        closed_loop_improvement_claimed=False,
    )
