"""Measured offline diagnostic labels with explicit local/retention separation."""

from dataclasses import dataclass
from typing import Any

import numpy as np

from rosclaw_soccer.rsi.postcontact_learning_observation import WIDTH, observation_contract
from rosclaw_soccer.rsi.postcontact_outcome_model import ACTIONS, require_observations


@dataclass(frozen=True)
class OutcomeDataset:
    features: np.ndarray
    labels: np.ndarray
    mask: np.ndarray
    context_indices: tuple[int, ...]
    source_clusters: tuple[int, ...]
    excluded_context_indices: tuple[int, ...]

    def local_indices(self) -> np.ndarray:
        return np.flatnonzero(np.asarray(self.context_indices) < 12)

    def retention_indices(self) -> np.ndarray:
        return np.flatnonzero(np.asarray(self.context_indices) >= 12)

    def group_folds(self) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
        local = self.local_indices()
        groups = np.asarray(self.source_clusters)
        unique = np.unique(groups[local])
        if len(local) < 4 or len(unique) < 2:
            raise ValueError("at least four observed local states in two source clusters required")
        # All heading/perturbation derivatives of one original course stay together.
        return tuple((local[groups[local] != g], local[groups[local] == g]) for g in unique)


def measured_dataset(readback: dict[str, Any]) -> OutcomeDataset:
    if (
        readback.get("schema") != "soccer.rsi.postcontact_full_bank_readback.v1"
        or readback.get("observation_contract") != observation_contract()
        or len(readback.get("candidate_supervision_not_admitted", [])) != 18
        or readback.get("training_authorized") is not False
        or readback.get("teacher_qualified") is not False
    ):
        raise ValueError("whole eighteen-context unqualified measured bank required")
    xs, ys, masks, indices, clusters, excluded = [], [], [], [], [], []
    for ci, row in enumerate(readback["candidate_supervision_not_admitted"]):
        expected = "CONSUMED_LOCAL_DEVELOPMENT" if ci < 12 else "CONSUMED_ORIGINAL_RETENTION"
        if (
            type(row.get("context_index")) is not int
            or row["context_index"] != ci
            or row.get("partition") != expected
            or type(row.get("source_cluster")) is not int
            or not 0 <= row["source_cluster"] < 3
            or row.get("training_authorized") is not False
            or type(row.get("observation_available")) is not bool
        ):
            raise ValueError("ordered original local/retention partitions and clusters required")
        boolean_fields = (
            "candidate_available",
            "observed_action_task_pass",
            "observed_action_safe",
            "observed_action_quality_measured",
        )
        if any(
            type(row.get(k)) is not list
            or len(row[k]) != ACTIONS
            or any(type(v) is not bool for v in row[k])
            for k in boolean_fields
        ):
            raise ValueError("all thirteen primitive action measurement flags required")
        passed = np.asarray(row["observed_action_task_pass"], dtype=bool)
        safe = np.asarray(row["observed_action_safe"], dtype=bool)
        quality = np.asarray(row["observed_action_quality_measured"], dtype=bool)
        available = np.asarray(row["candidate_available"], dtype=bool)
        if np.any(passed & (~safe | ~quality)):
            raise ValueError("passing labels must be safe and actually quality-measured")
        if not row["observation_available"]:
            if row["current_features"] is not None or available.any():
                raise ValueError("unobserved decisions cannot supply features or action authority")
            excluded.append(ci)
            continue
        if not available[0] or type(row.get("observed_trigger_frame")) is not int:
            raise ValueError("actual eligible observation and available baseline required")
        x = np.asarray(row["current_features"], dtype=np.float32)
        require_observations(x[None])
        xs.append(x)
        ys.append(np.stack((passed, safe), axis=1))
        # Right-censored quality is not a measured failure target for the pass head.
        masks.append(np.stack((available & quality, available), axis=1))
        indices.append(ci)
        clusters.append(row["source_cluster"])
    if not xs:
        raise ValueError("no observed current-state diagnostic training data")
    result = OutcomeDataset(
        np.asarray(xs, dtype=np.float32).reshape(-1, WIDTH),
        np.asarray(ys, dtype=bool),
        np.asarray(masks, dtype=bool),
        tuple(indices),
        tuple(clusters),
        tuple(excluded),
    )
    result.group_folds()
    return result


def masked_brier(
    probabilities: np.ndarray, labels: np.ndarray, mask: np.ndarray
) -> list[float | None]:
    if (
        probabilities.shape != labels.shape
        or labels.shape != mask.shape
        or probabilities.ndim != 3
        or probabilities.shape[1:] != (ACTIONS, 2)
        or not np.isfinite(probabilities).all()
        or np.any(probabilities < 0)
        or np.any(probabilities > 1)
        or labels.dtype != np.bool_
        or mask.dtype != np.bool_
    ):
        raise ValueError("finite whole measured success/safety probabilities and masks required")
    return [
        float(
            np.mean((probabilities[:, :, h][mask[:, :, h]] - labels[:, :, h][mask[:, :, h]]) ** 2)
        )
        if np.any(mask[:, :, h])
        else None
        for h in range(2)
    ]
