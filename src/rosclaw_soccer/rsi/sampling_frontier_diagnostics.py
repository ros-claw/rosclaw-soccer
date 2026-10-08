"""Offline outcome diagnostics, never a runtime selector or promotion gate."""

from __future__ import annotations

import math
from collections import Counter
from typing import Any

import numpy as np

from rosclaw_soccer.rsi.online_motor_actor_critic import terminal_return
from rosclaw_soccer.rsi.step_motor_features import FEATURE_NAMES


def motor_observation_boundaries(context: Any) -> dict[str, Any]:
    """Profile the original normalized135 observation, not a new input law.

    Exact values at +/-8 are recorded boundary hits, not proof that the raw
    values exceeded the limit. Caller authenticates the recorded learning
    sequence and its source normalizer; this function does not reconstruct
    normalization, physical dynamics, or information lost before clipping.
    """
    values = np.asarray(context)
    if (
        values.shape != (270, 135)
        or values.dtype != np.float64
        or not np.isfinite(values).all()
        or np.any(np.abs(values[:, :134]) > 8)
        or not np.isin(values[:, 134], (0, 1, 2)).all()
        or np.any(np.diff(values[:, 134]) < 0)
    ):
        raise ValueError("complete original finite normalized270x135 sequence required")
    rows = []
    for index, name in enumerate(FEATURE_NAMES):
        column = values[:, index]
        rows.append(
            {
                "feature_name": name,
                "positive_bound_frames": int(np.count_nonzero(column == 8)),
                "negative_bound_frames": int(np.count_nonzero(column == -8)),
                "minimum_recorded_value": float(np.min(column)),
                "maximum_recorded_value": float(np.max(column)),
            }
        )
    return {
        "schema": "soccer.rsi.offline_motor_observation_boundaries.v1",
        "feature_names": list(FEATURE_NAMES),
        "normalized_feature_count": 134,
        "control_frames": 270,
        "feature_coordinate_rows": 270 * 134,
        "recorded_bound": 8.0,
        "frames_with_any_boundary": int(
            np.count_nonzero(np.any(np.abs(values[:, :134]) == 8, axis=1))
        ),
        "features": rows,
        "contact_phase_counts": {
            str(p): int(np.count_nonzero(values[:, 134] == p)) for p in range(3)
        },
        "raw_normalization_reconstructed": False,
        "clipped_information_loss_proven": False,
        "source_physics_validated_here": False,
        "runtime_selection_authorized": False,
        "training_authorized": False,
        "promotion_authorized": False,
        "hardware_authorized": False,
    }


def contact_timeline(forces: Any, outcome: dict[str, Any]) -> dict[str, Any]:
    """Offline measured event labels; not future runtime observations.

    Each input is the maximum force norm during a control frame, not a
    substep force integral. Counts therefore describe active control frames,
    never contact duration or impulse. Caller must verify the archived trace
    and independent physical review before using these diagnostic labels.
    """
    values = np.asarray(forces)
    if (
        values.shape != (300, 6)
        or values.dtype.kind not in "fiu"
        or not np.isfinite(values).all()
        or np.any(values < 0)
    ):
        raise ValueError("complete finite 300-frame six-body force trace required")
    touched = np.flatnonzero(np.any(values > 1, axis=1))
    nonfoot = np.flatnonzero(np.any(values[:, 2:] > 1, axis=1))
    first = int(touched[0]) if len(touched) else None
    first_nonfoot = int(nonfoot[0]) if len(nonfoot) else None
    bodies = np.flatnonzero(np.max(values, axis=0) > 1).tolist()
    if (
        first != outcome.get("first_contact_frame")
        or bodies != outcome.get("contact_body_indices")
        or type(outcome.get("clean_foot_only")) is not bool
        or outcome["clean_foot_only"] != bool(touched.size and not nonfoot.size)
    ):
        raise ValueError("measured force events contradict original contact review")
    kind = (
        "no_contact"
        if first is None
        else "foot_only_all_episode"
        if first_nonfoot is None
        else "first_event_nonfoot"
        if first_nonfoot == first
        else "foot_first_then_nonfoot"
    )
    body_rows = []
    for body in range(6):
        active = np.flatnonzero(values[:, body] > 1)
        peak = int(np.argmax(values[:, body]))
        body_rows.append(
            dict(
                body_index=body,
                first_contact_frame=int(active[0]) if active.size else None,
                last_contact_frame=int(active[-1]) if active.size else None,
                active_control_frames=int(active.size),
                peak_frame_force_norm_n=float(values[peak, body]),
                peak_force_frame=peak if values[peak, body] > 0 else None,
            )
        )
    return dict(
        kind=kind,
        first_contact_frame=first,
        first_nonfoot_frame=first_nonfoot,
        secondary_nonfoot_lag_frames=(
            first_nonfoot - first
            if first_nonfoot is not None and first is not None and kind == "foot_first_then_nonfoot"
            else None
        ),
        per_body=body_rows,
        original_contact_threshold_n=1.0,
        active_frames_are_not_contact_duration=True,
        force_impulse_reconstructed=False,
        source_physics_validated_here=False,
        selection_uses_future_outcome_labels=True,
        runtime_selection_authorized=False,
        promotion_authorized=False,
        hardware_authorized=False,
    )


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
    rows: list[dict[str, Any]] = []
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
