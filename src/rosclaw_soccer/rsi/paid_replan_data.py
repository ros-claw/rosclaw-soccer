"""Admit executed expert rows only after ALL current-student CEM trials are audited."""

from typing import Any

import numpy as np

from rosclaw_soccer.rsi.paid_intervention_replan import replan_config
from rosclaw_soccer.rsi.paid_interventional_data import append_unique, executed_features


def admitted_replan_windows(
    producer: dict[str, Any], audit: dict[str, Any], producer_hash: str
) -> list[dict[str, Any]]:
    config = replan_config(104, 289)
    if (
        producer.get("status") != "PAID_INTERVENTIONAL_OWN_PREFIX_REPLAN_CLOSED_PENDING_AUDIT"
        or audit.get("status") != "PAID_INTERVENTIONAL_OWN_PREFIX_ALL_TRIAL_NN_IK_NATIVE_AUDIT_PASS"
        or audit.get("source_hash") != producer_hash
        or producer.get("config") != config
        or producer["counts"]["episodes"] != config["max_episodes"]
        or producer["counts"]["CEM_updates"] != 4
        or audit["counts"]["episodes"] != config["max_episodes"]
        or audit["counts"]["native"] != producer["actual_native_steps"]
        or audit["counts"]["neural"] != producer["actual_neural_rows"]
        or audit["counts"]["R2_planner"] != producer["counts"]["R2_planner"]
        or audit["counts"]["IK"] != producer["counts"]["IK"]
        or any(
            v.get("optimizer_updates") != 0 or v.get("Fresh_rows_read") != 0
            for v in (producer, audit)
        )
        or any(
            v.get(k) is not False
            for v in (producer, audit)
            for k in ("hardware_authorized", "promotion_authorized")
        )
    ):
        raise ValueError("complete all-trial same-source TRAIN-only independent witness required")
    if [s["switch_control"] for s in producer["searches"]] != config["switch_controls"] or [
        s["switch_control"] for s in audit["summaries"]
    ] != config["switch_controls"]:
        raise ValueError("both declared recovery windows required")
    admitted = []
    for search, summary in zip(producer["searches"], audit["summaries"], strict=True):
        cut = search["switch_control"]
        if len(search["selected"]) != 2 or len(summary["outcomes"]) != 2:
            raise ValueError("two fully audited selected repeats required")
        for rows in search["selected"] + summary["outcomes"]:
            if [r["case"] for r in rows] != [4, 5]:
                raise ValueError("complete paired source cases required")
        for index, row in enumerate(search["selected"][0]):
            outcomes = [rep[index] for rep in summary["outcomes"]]
            for repeat, own in enumerate(outcomes):
                recorded = search["selected"][repeat][index]
                healthy = all(
                    own.get(k) is True for k in ("quality", "stand_healthy", "stand_stable")
                )
                if (
                    recorded.get("healthy_success") is not healthy
                    or own.get("healthy_success") is not healthy
                    or recorded.get("quality") is not own.get("quality")
                    or recorded["switch_control"] != cut
                    or recorded["latent"] != search["latent"]
                ):
                    raise ValueError("actual expert contact and recovery witness differs")
            if all(o["healthy_success"] is True for o in outcomes):
                admitted.append(
                    dict(
                        case=row["case"],
                        switch_control=cut,
                        record=row,
                        canonical_repeat=0,
                        actual_physical_repeats=2,
                    )
                )
    if not admitted:
        raise ValueError("no independently successful current-state expert windows")
    return admitted


def aggregate_replan(
    parent: Any,
    previous: dict[str, np.ndarray],
    windows: list[dict[str, Any]],
    read: Any,
    *,
    source_hash: str,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    additions = []
    for window in windows:
        cut, row = window["switch_control"], window["record"]
        raw = read(row["kick"])
        z, frames = executed_features(parent, raw, cut)
        additions.append(
            (
                z,
                frames,
                dict(
                    source_hash=source_hash,
                    case=window["case"],
                    switch_control=cut,
                    canonical_repeat=0,
                    physical_repeats=2,
                    raw_path=row["kick"]["raw_path"],
                    raw_hash=row["kick"]["raw_hash"],
                    whole_source_rows=len(raw["NN_action"]),
                    student_prefix_rows_excluded=cut,
                ),
            )
        )
    return append_unique(previous, additions)
