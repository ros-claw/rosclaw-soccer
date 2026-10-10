"""Explain complete paired consumed contact outcomes, never certify a policy.

Inputs are caller-supplied primitive measurements, not authenticated receipts.
The caller must verify source receipts and physical replay separately. Keep
every case, including regressions, rather than reporting only net successes.
"""

from __future__ import annotations

import math
from typing import Any


def _finite(value: Any) -> float:
    if not isinstance(value, (int, float)) or type(value) not in (int, float):
        raise ValueError("ordinary numeric physical measurements required")
    if abs(value) > 1e6 or not math.isfinite(value):
        raise ValueError("bounded finite physical measurements required")
    return float(value)


def _labels(value: Any) -> tuple[str, ...]:
    required = {
        "first_contact_frame",
        "clean_foot_only",
        "high_quality",
        "maximum_lateral_excursion_m",
        "minimum_pelvis_z_m",
        "forward_60_m",
        "lateral_over_forward_60",
    }
    if (
        type(value) is not dict
        or not required <= value.keys()
        or any(type(value.get(key)) is not bool for key in ("clean_foot_only", "high_quality"))
    ):
        raise ValueError("ordinary boolean physical outcome labels required")
    first = value.get("first_contact_frame")
    if first is not None and (type(first) is not int or not 0 <= first < 300):
        raise ValueError("bounded first contact frame or explicit absence required")
    for key in ("maximum_lateral_excursion_m", "minimum_pelvis_z_m"):
        _finite(value.get(key))
    if value["maximum_lateral_excursion_m"] < 0:
        raise ValueError("nonnegative maximum excursion required")
    forward, ratio = value.get("forward_60_m"), value.get("lateral_over_forward_60")
    complete = first is not None and first + 60 < 300
    if complete:
        if _finite(ratio) < 0:
            raise ValueError("nonnegative absolute lateral ratio required")
        _finite(forward)
    elif forward is not None or ratio is not None:
        raise ValueError("unobserved post-contact window must remain absent")
    if first is None and value["clean_foot_only"]:
        raise ValueError("clean foot contact requires an actual contact")
    labels = []
    if first is None:
        labels.append("NO_CONTACT")
    elif not value["clean_foot_only"]:
        labels.append("NONFOOT_CONTACT")
    if not complete:
        labels.append("NO_COMPLETE_POST_CONTACT_WINDOW")
    else:
        if _finite(forward) < 1:
            labels.append("INSUFFICIENT_FORWARD_DISPLACEMENT")
        if _finite(ratio) > 0.3:
            labels.append("EXCESS_LATERAL_RATIO")
    if value["maximum_lateral_excursion_m"] > 4:
        labels.append("OUT_OF_PLAY")
    # Stability is a separate safety condition, not part of the historical HQ
    # definition. Never silently change the exam by folding it into HQ.
    if value["high_quality"] is not (not labels):
        raise ValueError("high-quality claim differs from original physical thresholds")
    if value["minimum_pelvis_z_m"] < 0.65:
        labels.append("PELVIS_SAFETY_FAILURE")
    return tuple(labels)


def paired_contact_failure_map(
    parent: list[dict[str, Any]], candidate: list[dict[str, Any]]
) -> dict[str, Any]:
    """Explain all52 cases; case indices must align and cannot be dropped.

    Thresholds exactly match the consumed CPU exam. This is descriptive
    failure mining, not fresh evaluation, causal diagnosis or training admission.
    """
    if any(type(rows) is not list or len(rows) != 52 for rows in (parent, candidate)):
        raise ValueError("both complete ordered52-case outcomes required")
    records: list[dict[str, Any]] = []
    for index, (old, new) in enumerate(zip(parent, candidate, strict=True)):
        if any(
            type(row) is not dict
            or type(row.get("index")) is not int
            or row["index"] != index
            or type(row.get("outcome")) is not dict
            for row in (old, new)
        ):
            raise ValueError("exact aligned case indices0..51 required")
        before, after = old["outcome"], new["outcome"]
        old_labels, new_labels = _labels(before), _labels(after)
        first_old, first_new = before["first_contact_frame"], after["first_contact_frame"]
        records.append(
            dict(
                index=index,
                parent_failures=list(old_labels),
                candidate_failures=list(new_labels),
                introduced_failures=[label for label in new_labels if label not in old_labels],
                resolved_failures=[label for label in old_labels if label not in new_labels],
                high_quality_lost=before["high_quality"] and not after["high_quality"],
                high_quality_gained=after["high_quality"] and not before["high_quality"],
                first_contact_shift_frames=(
                    first_new - first_old
                    if first_old is not None and first_new is not None
                    else None
                ),
            )
        )
    return dict(
        schema="soccer.rsi.paired_consumed_contact_failure_map.v1",
        compared_case_count=52,
        records=records,
        high_quality_lost_cases=[r["index"] for r in records if r["high_quality_lost"]],
        high_quality_gained_cases=[r["index"] for r in records if r["high_quality_gained"]],
        new_out_of_play_cases=[
            r["index"] for r in records if "OUT_OF_PLAY" in r["introduced_failures"]
        ],
        input_receipts_authenticated_here=False,
        physical_replay_performed_here=False,
        causal_failure_reason_proven=False,
        training_data_admitted=False,
        fresh_gain_verified=False,
        promotion_authorized=False,
        hardware_authorized=False,
    )
