"""Offline credit inventory for the current 30..299-frame contact learner.

This is a football adapter, not an optimizer or an outcome-conditioned actor.
The caller must bind the complete records and numeric array to independently
verified physical receipts. No curriculum or policy is activated here.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from rosclaw_soccer.rsi.sampling_frontier_diagnostics import sampling_frontier


def sampling_credit_inventory(
    records: list[dict[str, Any]], advantages: Any, *, expected_episodes: int
) -> dict[str, Any]:
    """Locate signed credit without discarding failures or changing rewards.

    Failure strata overlap; contact-time strata partition all frame samples.
    Positive credit on a failed episode is not necessarily an error: these are
    relative advantages, not success labels, nor a causal attribution proof.
    """
    if (
        type(expected_episodes) is not int
        or not 1 <= expected_episodes <= 4096
        or len(records) != expected_episodes
        or [r.get("group") for r in records] != list(range(expected_episodes))
    ):
        raise ValueError("complete ordered declared episodes, including failures, required")
    frontier = sampling_frontier(records, expected_episodes=expected_episodes)
    values = np.asarray(advantages)
    if (
        values.dtype != np.float64
        or values.shape != (expected_episodes, 270)
        or not np.isfinite(values).all()
    ):
        raise ValueError("complete finite float64 30..299-frame advantages required")

    def describe(selected: np.ndarray[Any, Any]) -> dict[str, Any]:
        with np.errstate(over="ignore", invalid="ignore"):
            total = float(selected.sum())
            absolute = float(np.abs(selected).sum())
        if not np.isfinite([total, absolute]).all():
            raise ValueError("finite credit totals required")
        return dict(
            frame_samples=int(selected.size),
            positive_rows=int(np.sum(selected > 0)),
            negative_rows=int(np.sum(selected < 0)),
            zero_rows=int(np.sum(selected == 0)),
            signed_credit_sum=total,
            absolute_credit_sum=absolute,
            mean_credit=total / selected.size if selected.size else None,
        )

    families: dict[str, list[int]] = {"high_quality": [], "not_high_quality": []}
    before = np.zeros(values.shape, dtype=bool)
    after = np.zeros(values.shape, dtype=bool)
    no_contact = np.zeros(values.shape, dtype=bool)
    frames = np.arange(30, 300)
    for index, record in enumerate(records):
        outcome = record["outcome"]
        families["high_quality" if outcome["high_quality"] else "not_high_quality"].append(index)
        # Reuse the unchanged measured HQ/reward checks, including dirty and
        # unsafe trajectories; do not silently invent a new success definition.
        one = sampling_frontier([record], expected_episodes=expected_episodes)
        for reason in one["overlapping_failure_counts"]:
            families.setdefault(reason, []).append(index)
        first = outcome["first_contact_frame"]
        if first is None:
            no_contact[index] = True
        else:
            before[index] = frames < first
            after[index] = frames >= first
    if not np.all(before.astype(int) + after.astype(int) + no_contact.astype(int) == 1):
        raise ValueError("contact-time strata must partition every training sample")
    strata = {
        name: dict(episode_groups=groups, **describe(values[groups]))
        for name, groups in sorted(families.items())
    }
    return dict(
        schema="soccer.rsi.offline_sampling_credit_inventory.v1",
        completed_episodes=expected_episodes,
        frame_samples=expected_episodes * 270,
        training_frame_range=[30, 299],
        all_success_and_failure_rows_included=True,
        aggregate=describe(values),
        overlapping_outcome_strata=strata,
        contact_time_strata={
            "before_first_contact": describe(values[before]),
            "at_or_after_first_contact": describe(values[after]),
            "episode_without_contact": describe(values[no_contact]),
        },
        measured_frontier=frontier,
        positive_failed_credit_is_not_necessarily_wrong=True,
        selection_uses_future_outcome_labels=True,
        causal_credit_assignment_proven=False,
        source_physics_validated_here=False,
        runtime_selection_authorized=False,
        training_authorized=False,
        promotion_authorized=False,
        hardware_authorized=False,
        qualification="OFFLINE_CREDIT_DIAGNOSTIC_NOT_POLICY_GAIN",
    )
