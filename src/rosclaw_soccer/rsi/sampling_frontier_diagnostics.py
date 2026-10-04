"""Offline outcome diagnostics, never a runtime selector or promotion gate."""

from __future__ import annotations

import math
from collections import Counter
from typing import Any

from rosclaw_soccer.rsi.online_motor_actor_critic import terminal_return


def sampling_frontier(records: list[dict[str, Any]], *, expected_episodes: int) -> dict[str, Any]:
    """Keep failures and successful donor indices without claiming a deployed union.

    Caller must independently validate source records and their physical receipts.
    Partial arrival order is not random: no confidence interval or extrapolation.
    Failure reasons overlap deliberately, so their sum need not equal episodes.
    """
    if type(expected_episodes) is not int or expected_episodes < 1:
        raise ValueError("positive declared episode count required")
    if not records or len(records) > expected_episodes:
        raise ValueError("nonempty bounded completed records required")
    seen: set[int] = set()
    contexts: dict[int, list[dict[str, Any]]] = {}
    reasons: Counter[str] = Counter()
    for record in records:
        group = record.get("group")
        context = record.get("baseline_course_index")
        if (
            type(group) is not int
            or not 0 <= group < expected_episodes
            or group in seen
            or type(context) is not int
            or context < 0
        ):
            raise ValueError("unique declared episode and original context required")
        seen.add(group)
        outcome = record["outcome"]
        terminal_return(outcome)  # Preserve the actual training objective.
        bodies = outcome.get("contact_body_indices")
        first = outcome.get("first_contact_frame")
        if (
            not isinstance(bodies, list)
            or any(type(body) is not int or body not in range(6) for body in bodies)
            or (first is not None and (type(first) is not int or not 0 <= first < 300))
        ):
            raise ValueError("measured contact event and body identities required")
        clean = first is not None and bool(bodies) and set(bodies) <= {0, 1}
        if clean != outcome["clean_foot_only"]:
            raise ValueError("contact evidence contradicts clean-foot flag")
        forward, lateral = outcome.get("forward_60_m"), outcome.get("lateral_60_m")
        if clean and any(
            type(v) not in (int, float) or not math.isfinite(v) for v in (forward, lateral)
        ):
            raise ValueError("measured post-contact displacement required")
        safe = outcome["minimum_pelvis_z_m"] >= 0.65
        in_play = outcome["maximum_lateral_excursion_m"] <= 4
        measured_hq = bool(clean and in_play and forward >= 1 and abs(lateral) / forward <= 0.3)
        if measured_hq != outcome["high_quality"]:
            raise ValueError("physical measurements contradict high-quality flag")
        if not safe:
            reasons["unsafe_pelvis"] += 1
        if not in_play:
            reasons["out_of_play"] += 1
        if first is None or not bodies:
            reasons["no_contact"] += 1
        elif not clean:
            reasons["nonfoot_contact"] += 1
        else:
            if forward < 1:
                reasons["insufficient_forward"] += 1
            if abs(lateral) / max(forward, 0.01) > 0.3:
                reasons["direction_error"] += 1
        contexts.setdefault(context, []).append(record)
    rows = []
    for context, samples in sorted(contexts.items()):
        successful = [r for r in samples if r["outcome"]["high_quality"]]
        rows.append(
            dict(
                baseline_course_index=context,
                completed_episodes=len(samples),
                high_quality_episodes=len(successful),
                successful_donor_groups=sorted(r["group"] for r in successful),
                maximum_observed_terminal_return=max(
                    terminal_return(r["outcome"]) for r in samples
                ),
            )
        )
    return dict(
        schema="soccer.rsi.offline_sampling_frontier.v1",
        completed_episodes=len(records),
        expected_episodes=expected_episodes,
        complete=len(records) == expected_episodes,
        observed_high_quality_episodes=sum(r["high_quality_episodes"] for r in rows),
        observed_contexts=len(rows),
        observed_successful_contexts=sum(r["high_quality_episodes"] > 0 for r in rows),
        overlapping_failure_counts=dict(sorted(reasons.items())),
        contexts=rows,
        selection_uses_future_outcome_labels=True,
        partial_results_extrapolated=False,
        source_physics_validated_here=False,
        runtime_selection_authorized=False,
        training_authorized=False,
        promotion_authorized=False,
        hardware_authorized=False,
        qualification="OFFLINE_DIAGNOSTIC_NOT_DEPLOYED_POLICY",
    )
