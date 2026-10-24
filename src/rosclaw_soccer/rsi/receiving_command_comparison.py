"""Paired consumed-TRAIN command learning review; never a promotion gate."""

import math
from collections import Counter
from typing import Any

from rosclaw_soccer.rsi.pass_transfer_learning import controlled_pass_label

INDICES = (0, 6, 24, 32, 60, 64)
SEEDS = (732023, 732024, 732025)
BC_SOURCE = "RECEIVING_NEURAL_COMMAND_BC_PILOT_PENDING_OWN"
BC_OWN = "RECEIVING_NEURAL_COMMAND_BC_PILOT_ALL_OWN_MATCH"
TASK_SOURCE = "RECEIVING_TASK_COMMAND_DAGGER_PILOT_PENDING_OWN"
TASK_OWN = "RECEIVING_TASK_COMMAND_DAGGER_PILOT_ALL_OWN_MATCH"


def _digest(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 71
        and value.startswith("sha256:")
        and all(c in "0123456789abcdef" for c in value[7:])
    )


def admit_pair(
    source: dict[str, Any],
    own: dict[str, Any],
    *,
    source_hash: str,
    training_hash: str,
    seed: int,
    task: bool,
) -> list[dict[str, Any]]:
    """Require a complete exact independent manual-clock replay, not its claim alone.

    Callers authenticate file bytes and inherited pins before invoking this pure
    boundary. Paired repeats remain executions, never independent course counts.
    """
    if (
        seed not in SEEDS
        or type(seed) is not int
        or type(task) is not bool
        or not _digest(source_hash)
        or not _digest(training_hash)
        or source.get("status") != (TASK_SOURCE if task else BC_SOURCE)
        or own.get("status") != (TASK_OWN if task else BC_OWN)
        or source.get("source_hash") is not None
        or own.get("source_hash") != source_hash
        or source.get("independent_whole_world_replay") is not False
        or own.get("independent_whole_world_replay") is not True
    ):
        raise ValueError("exact closed source/own command pilot required")
    for report in (source, own):
        if (
            report.get("command_training_hash") != training_hash
            or report.get("command_seed") != seed
            or report.get("source_case_indices") != list(INDICES)
            or not isinstance(report.get("source_head"), str)
            or len(report["source_head"]) != 40
            or any(
                report.get(k) is not False
                for k in (
                    "Fresh_verified",
                    "promotion_authorized",
                    "hardware_authorized",
                    "goal_complete",
                )
            )
            or report.get("optimizer_updates") != 0
            or report.get("all_actual_native_zero_identity") is not False
            or (task and report.get("command_stage") != "task-dagger-v2")
            or (not task and report.get("command_reference_untrained", False) is not False)
            or type(report.get("records")) is not list
            or len(report["records"]) != 12
        ):
            raise ValueError("fixed trained command cohort binding required")
    if source["source_head"] != own["source_head"] or any(
        source.get(k) != own.get(k) or not _digest(source.get(k))
        for k in ("initialization_hash", "training_hash")
    ):
        raise ValueError("source/own actor and motor ancestry must agree")
    records: list[dict[str, Any]] = source["records"]
    if own["records"] != [{k: v for k, v in row.items() if k != "artifacts"} for row in records]:
        raise ValueError("all independent physical and task records must match")
    for i, row in enumerate(records):
        controls = row.get("controls")
        if (
            row.get("index") != i
            or row.get("source_case_index") != INDICES[i // 2]
            or type(controls) is not int
            or not 100 < controls <= 400
            or row.get("native_steps") != controls * 10
            or row.get("actor_calls") != controls * 5
            or row.get("observed_recovery_controls") != 100
            or row.get("all_actor_actions_actually_executed") is not True
            or row.get("receiver_command_own_body_and_actual_feedback") is not True
            or row.get("receiver_command_teacher_actions_executed") is not False
            or row.get("receiver_command_updates_during_episode") != 0
            or row.get("assisted") is not False
            or row.get("external_teacher") is not False
            or row.get("receiver_motor_optimizer_updates") != 0
            or row.get("controller_kind")
            != (
                "receiving-neural-task-command-dagger-native-v2"
                if task
                else "receiving-neural-command-bc-native-v1"
            )
            or row.get("A_recovery", {}).get("controls") != 100
            or row.get("A_recovery", {}).get("neural_rows") != 100
            or row.get("A_recovery", {}).get("native_steps") != 1000
            or row.get("outcome", {}).get("formal_native_steps") != 3000
        ):
            raise ValueError("actual fixed-clock five-network full recovery pilot required")
        out = row["outcome"]
        for key in (
            "B_legal_functional_receive_before_deadline",
            "combined_precision_and_functional_receive",
            "whole_body_healthy",
            "passer_clean_foot",
            "passer_full100NNrecovery",
            "B_excluded_contact",
        ):
            if type(out.get(key)) is not bool:
                raise ValueError("typed actual physical outcomes required")
        _metrics(out)
    clusters = Counter(row["artifacts"]["initial"]["hash"] for row in records)
    if sorted(clusters.values()) != [2, 4, 6]:
        raise ValueError(
            "three frozen physical initial clusters with paired 6/4/2 executions required"
        )
    counts = dict(
        native_steps=sum(r["native_steps"] for r in records),
        actor_calls=sum(r["actor_calls"] for r in records),
    )
    if source.get("counts") != counts or own.get("counts") != counts:
        raise ValueError("all source and own executed steps must be charged")
    return records


def _metrics(out: dict[str, Any]) -> dict[str, bool]:
    controlled = controlled_pass_label(out)
    error = out["incoming"]["incoming_target_error_3d_before_B_contact_m"]
    speed = out["incoming_arrival_speed_mps"]
    if any(
        v is not None and (type(v) not in (int, float) or not math.isfinite(v) or v < 0)
        for v in (error, speed)
    ) or (error is None) != (speed is None):
        raise ValueError("paired finite physical precision and speed measurements required")
    strict = bool(
        controlled and error is not None and error <= 0.1 and speed is not None and speed <= 3.0
    )
    if out["combined_precision_and_functional_receive"] is not strict:
        raise ValueError("unchanged .1m/3mps controlled-receive strict label required")
    return dict(
        functional=out["B_legal_functional_receive_before_deadline"],
        controlled=controlled,
        strict=strict,
        healthy=out["whole_body_healthy"],
        clean_foot=out["passer_clean_foot"],
        full_recovery=out["passer_full100NNrecovery"],
        legal_contact=not out["B_excluded_contact"],
    )


def compare_records(parent: list[dict[str, Any]], child: list[dict[str, Any]]) -> dict[str, Any]:
    """Compare already admitted matched worlds without changing frozen scoring."""
    if len(parent) != 12 or len(child) != 12:
        raise ValueError("complete admitted cohorts required")
    for a, b in zip(parent, child, strict=True):
        if any(
            a.get(k) != b.get(k) for k in ("index", "source_case_index", "course", "sampling_seed")
        ):
            raise ValueError("paired identical course and private A motor noise required")
        ia = a.get("artifacts", {}).get("initial", {}).get("hash")
        ib = b.get("artifacts", {}).get("initial", {}).get("hash")
        if not _digest(ia) or ia != ib:
            raise ValueError("actual identical physical initial artifacts required")
    before = [_metrics(r["outcome"]) for r in parent]
    after = [_metrics(r["outcome"]) for r in child]
    metrics: dict[str, dict[str, Any]] = {}
    for name in before[0]:
        old = {i for i, m in enumerate(before) if m[name]}
        new = {i for i, m in enumerate(after) if m[name]}
        metrics[name] = dict(
            parent=len(old),
            child=len(new),
            rescues=sorted(new - old),
            regressions=sorted(old - new),
        )
    loss = any(m["regressions"] for m in metrics.values())
    better = metrics["strict"]["child"] > metrics["strict"]["parent"]
    return dict(
        worlds=12,
        distinct_course_indices=6,
        paired_repeats_are_independent_contexts=False,
        metrics=metrics,
        local_strict_gain_without_any_measured_loss=better and not loss,
        next_agent_action=(
            "EXPAND_DISTINCT_COURSES_BEFORE_FRESH_OR_TEAM_CLAIMS"
            if better and not loss
            else "DIAGNOSE_OWN_STATE_FAILURES_AND_RETENTION_NO_BLIND_EPOCHS"
        ),
        Fresh_verified=False,
        promotion_authorized=False,
        goal_complete=False,
    )
