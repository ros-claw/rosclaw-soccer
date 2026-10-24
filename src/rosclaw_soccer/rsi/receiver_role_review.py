"""All96 same-noise role interventions, with reception retention and clustering.

No physics, fitting, reward changes, significance from duplicated frames,
Fresh access, or promotion. Known selected rescues are reported separately.
"""

import math
from collections import Counter
from typing import Any

from rosclaw_soccer.rsi.parallel_refinement_batch import (
    FULL_OWN,
    FULL_SOURCE,
    ROLE_OWN,
    ROLE_SOURCE,
    parallel_config,
    verify_parallel_batch,
)
from rosclaw_soccer.rsi.pass_transfer_learning import controlled_pass_label


def _metrics(o: dict[str, Any]) -> dict[str, Any]:
    controlled = controlled_pass_label(o)
    error = o["incoming"]["incoming_target_error_3d_before_B_contact_m"]
    speed = o["incoming_arrival_speed_mps"]
    if any(
        x is not None and (type(x) not in (int, float) or not math.isfinite(x) or x < 0)
        for x in (error, speed)
    ) or (error is None) != (speed is None):
        raise ValueError("actual finite paired delivery measurements required")
    strict = bool(
        controlled and error is not None and error <= 0.1 and speed is not None and speed <= 3.0
    )
    if (
        type(o["combined_precision_and_functional_receive"]) is not bool
        or o["combined_precision_and_functional_receive"] != strict
    ):
        raise ValueError("unchanged physical strict label must match all actual criteria")
    return dict(
        strict=strict,
        controlled=controlled,
        functional=o["B_legal_functional_receive_before_deadline"],
        healthy=o["whole_body_healthy"],
        clean_foot=o["passer_clean_foot"],
        full_recovery=o["passer_full100NNrecovery"],
        excluded_B=o["B_excluded_contact"],
        error_m=error,
        speed_mps=speed,
    )


def _effect(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = dict(worlds=len(rows))
    for metric in ("strict", "functional", "controlled"):
        result[metric + "_old"] = sum(r["old"][metric] for r in rows)
        result[metric + "_new"] = sum(r["new"][metric] for r in rows)
        result[metric + "_rescues"] = [
            r["index"] for r in rows if not r["old"][metric] and r["new"][metric]
        ]
        result[metric + "_regressions"] = [
            r["index"] for r in rows if r["old"][metric] and not r["new"][metric]
        ]
    result["safety_losses"] = [
        r["index"]
        for r in rows
        if any(r["old"][k] and not r["new"][k] for k in ("healthy", "clean_foot", "full_recovery"))
        or (not r["old"]["excluded_B"] and r["new"]["excluded_B"])
    ]
    return result


def paired_role_review(
    factual: dict[str, Any],
    factual_own: dict[str, Any],
    source: dict[str, Any],
    own: dict[str, Any],
    *,
    factual_source_hash: str,
    source_hash: str,
) -> dict[str, Any]:
    def pair(src: dict[str, Any], audit: dict[str, Any], role: bool, exact_hash: str) -> None:
        cfg = parallel_config(26, 32, full_recovery=True, receiver_role_ablation=role)
        if (
            src["config"] != cfg
            or audit["config"] != cfg
            or len(src["records"]) != 96
            or len(audit["records"]) != 96
        ):
            raise ValueError("exact fixed-A role or factual96 protocol required")
        courses = [src["records"][i]["course"] for i in (0, 32, 64)]
        verify_parallel_batch(src, courses)
        verify_parallel_batch(audit, courses)
        if (
            src["status"] != (ROLE_SOURCE if role else FULL_SOURCE)
            or audit["status"] != (ROLE_OWN if role else FULL_OWN)
            or audit["source_hash"] != exact_hash
            or audit["source_head"] != src["source_head"]
            or audit["training_hash"] != src["training_hash"]
            or audit["counts"] != src["counts"]
            or audit["independent_whole_world_replay"] is not True
            or audit["sampling_law_audit_passed"] is not True
            or audit["records"]
            != [{k: v for k, v in r.items() if k != "artifacts"} for r in src["records"]]
            or any(
                r.get(k) is not False
                for r in (src, audit)
                for k in (
                    "Fresh_verified",
                    "team_game_verified",
                    "actor_training_admitted",
                    "hardware_authorized",
                    "promotion_authorized",
                    "goal_complete",
                )
            )
            or any(
                type(r.get("optimizer_updates")) is not int or r["optimizer_updates"] != 0
                for r in (src, audit)
            )
        ):
            raise ValueError("all96 exact independently audited native/law histories required")

    pair(factual, factual_own, False, factual_source_hash)
    pair(source, own, True, source_hash)
    if source["training_hash"] != factual["training_hash"]:
        raise ValueError("A behavior must remain the identical checkpoint across B interventions")
    rows = []
    for i, (old, new) in enumerate(zip(factual["records"], source["records"], strict=True)):
        h = old["artifacts"]["initial"]["hash"]
        if (
            old["index"] != i
            or new["index"] != i
            or old["course"] != new["course"]
            or old["sampling_seed"] != new["sampling_seed"]
            or new["artifacts"]["initial"]["hash"] != h
        ):
            raise ValueError("exact paired initial states, goals and private seeds required")
        rows.append(
            dict(
                index=i,
                initial_hash=h,
                known_selected_intervention=i in (6, 24),
                old=_metrics(old["outcome"]),
                new=_metrics(new["outcome"]),
            )
        )
    groups = Counter(r["initial_hash"] for r in rows)
    if sorted(groups.values()) != [32, 32, 32]:
        raise ValueError("exact three physical initial clusters, not96 independent initials")
    overall = _effect(rows)
    known = _effect([r for r in rows if r["known_selected_intervention"]])
    remaining = _effect([r for r in rows if not r["known_selected_intervention"]])
    clusters = [
        dict(initial_hash=h, **_effect([r for r in rows if r["initial_hash"] == h])) for h in groups
    ]
    rescued_clusters = sum(bool(c["strict_rescues"]) for c in clusters)
    broader = bool(
        remaining["strict_rescues"]
        and rescued_clusters >= 2
        and not overall["safety_losses"]
        and not overall["controlled_regressions"]
        and not overall["functional_regressions"]
        and not overall["strict_regressions"]
    )
    if overall["safety_losses"]:
        action = "REJECT_SAFETY_REGRESSION_REVIEW_CAUSE"
    elif overall["strict_regressions"]:
        action = "REJECT_STRICT_REGRESSION_REVIEW_CAUSE"
    elif overall["controlled_regressions"] or overall["functional_regressions"]:
        action = "REPAIR_RECEIVING_RETENTION_BEFORE_LEARNING_OR_PROMOTION"
    elif not remaining["strict_rescues"]:
        action = "ONLY_KNOWN_LOCAL_RESCUES_EXPAND_DISTINCT_TRAIN_COURSES"
    elif rescued_clusters < 2:
        action = "LOCAL_ONE_INITIAL_SUPPORT_REQUIRES_DISTINCT_INITIAL_VALIDATION"
    else:
        action = "DESIGN_SEPARATE_ROLE_AWARE_LEARNING_ENVIRONMENT_AND_RETENTION_PROTOCOL"
    return dict(
        status="RECEIVER_ROLE_PAIRED96_REVIEWED_NOT_LEARNING_OR_PROMOTION",
        overall=overall,
        known_selected_2=known,
        remaining_94=remaining,
        initial_clusters=clusters,
        unique_consumed_TRAIN_initials=3,
        private_noise_seeds_per_initial=32,
        broader_consumed_TRAIN_role_support=broader,
        next_agent_action=action,
        rows=rows,
        physical_steps_executed=0,
        optimizer_updates=0,
        Fresh_verified=False,
        continuous_learning_proven=False,
        receiver_neural_learning_proven=False,
        team_game_verified=False,
        promotion_authorized=False,
        hardware_authorized=False,
        goal_complete=False,
    )
