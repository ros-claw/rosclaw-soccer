"""Paired SIM heading/exit consequences, not a deployed actor or teacher.

112-feature input is current frame30 body/ball/Foundation navigation target,
BEFORE heading, IK and residual. It is privileged SIM diagnostic data, not an
approved student perception contract. Future outcomes are LABELS only.
"""

from typing import Any

import numpy as np

from rosclaw_soccer.rsi.body_ball_response_features import measured_body_ball_features
from rosclaw_soccer.rsi.first_touch_outcome import label_first_touch
from rosclaw_soccer.rsi.foot_heading_evidence import FIELDS, PREFIX
from rosclaw_soccer.rsi.foot_heading_parity import original_arrays, require_heading_parity
from rosclaw_soccer.rsi.postcontact_release_control import ACTION_NAMES
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json

PAIRS = ((0, 2), (1, 3), (4, 6), (5, 7), (8, 10), (9, 11), (12, 13), (14, 15), (16, 17))
DECISION_FRAME = 30
RELEASE_FIELDS = {
    "postcontact_release_requested_offset_nm",
    "postcontact_release_applied_offset_nm",
    "postcontact_release_forecast_steps",
}


def before_heading_observation(trace: dict[str, np.ndarray], *, enabled: bool) -> np.ndarray:
    if type(enabled) is not bool:
        raise ValueError("explicit OFF/ON heading arm required")
    extra = {k for k in trace if k.startswith("postcontact_release_")}
    if extra != RELEASE_FIELDS:
        raise ValueError("exact complete fixed release capture required")
    for name, shape, dtype in (
        ("postcontact_release_requested_offset_nm", (300, 1, 29), np.float64),
        ("postcontact_release_applied_offset_nm", (300, 1, 10, 29), np.float64),
        ("postcontact_release_forecast_steps", (300, 1, 1), np.int64),
    ):
        value = trace[name]
        if (
            type(value) is not np.ndarray
            or value.shape != shape
            or value.dtype != dtype
            or not np.isfinite(value).all()
            or np.any(value[: DECISION_FRAME + 1])
            or (name.endswith("forecast_steps") and np.any(value))
        ):
            raise ValueError(
                "whole no-forecast release capture with zero pre-contact action required"
            )
    original = {k: v for k, v in trace.items() if k not in extra}
    original_arrays(original)
    if (
        {k for k in original if k.startswith(PREFIX)} != {PREFIX + k for k in FIELDS}
        or not np.all(original["foot_heading_state"][:, 0, 3] == int(enabled))
        or np.any(original["force_n"][:DECISION_FRAME] > 1)
    ):
        raise ValueError("same explicit heading arm and genuinely pre-contact decision required")
    result: np.ndarray = measured_body_ball_features(
        original["canonical_qpos"][DECISION_FRAME],
        original["canonical_qvel"][DECISION_FRAME],
        original["foundation_navigation_joint_target_rad"][DECISION_FRAME],
    )[0].copy()
    result.flags.writeable = False
    return result


def compare_pair(
    off: dict[str, np.ndarray], on: dict[str, np.ndarray]
) -> tuple[np.ndarray, dict[str, Any]]:
    x = before_heading_observation(off, enabled=False)
    other = before_heading_observation(on, enabled=True)
    a = {k: v for k, v in off.items() if k not in RELEASE_FIELDS}
    b = {k: v for k, v in on.items() if k not in RELEASE_FIELDS}
    parity = require_heading_parity(a, b, frames=DECISION_FRAME)
    for key in ("canonical_qpos", "canonical_qvel", "foundation_navigation_joint_target_rad"):
        if a[key][DECISION_FRAME].tobytes() != b[key][DECISION_FRAME].tobytes():
            raise ValueError("same body and Foundation target BEFORE heading decision required")
    if x.tobytes() != other.tobytes():
        raise ValueError("same current-state input for the two heading treatments required")
    return x, parity


def paired_context(
    off: dict[str, Any],
    on: dict[str, Any],
    traces: tuple[list[dict[str, np.ndarray]], list[dict[str, np.ndarray]]],
    *,
    index: int,
) -> dict[str, Any]:
    if type(index) is not int or not 0 <= index < len(PAIRS):
        raise ValueError("one of nine declared whole heading pairs required")
    oi, ni = PAIRS[index]
    partition = "CONSUMED_LOCAL_DEVELOPMENT" if index < 6 else "CONSUMED_ORIGINAL_RETENTION"
    names = [*ACTION_NAMES, "baseline"]
    if (
        off["context_index"] != oi
        or on["context_index"] != ni
        or off["partition"] != partition
        or on["partition"] != partition
        or any(type(c[k]) is not int for c in (off, on) for k in ("source_index", "context_index"))
        or off["source_index"] != 2 * (index // 2 if index < 6 else index - 6)
        or on["source_index"] != off["source_index"] + 1
        or any(len(c["rows"]) != 14 for c in (off, on))
        or len(traces) != 2
        or any(len(t) != 14 for t in traces)
    ):
        raise ValueError("whole original OFF/ON contexts and their fourteen actual traces required")
    x = None
    outcomes: list[list[dict[str, Any]]] = [[], []]
    parities = []
    for ai, name in enumerate(names):
        left, right = off["rows"][ai], on["rows"][ai]
        if (
            left["context_index"] != oi
            or right["context_index"] != ni
            or left["action_index"] != ai
            or right["action_index"] != ai
            or left["action_name"] != name
            or right["action_name"] != name
            or left["course"] != right["course"]
            or left["course"] != off["rows"][0]["course"]
            or left["original_policy_hash"] != right["original_policy_hash"]
            or left["original_policy_hash"] != off["rows"][0]["original_policy_hash"]
        ):
            raise ValueError("same course/policy and exact matched named interventions required")
        current, parity = compare_pair(traces[0][ai], traces[1][ai])
        if x is None:
            x = current
        if current.tobytes() != x.tobytes():
            raise ValueError("all28 treatments must share one predecision observation")
        parities.append(parity)
        for heading, row in enumerate((left, right)):
            outcome = label_first_touch(traces[heading][ai]).to_dict()
            if outcome != row["outcome"]:
                raise ValueError("independently relabeled whole original M0 outcome required")
            if ai < 13:
                outcomes[heading].append(outcome)
    assert x is not None
    result = dict(
        schema="soccer.rsi.precontact_heading_pair.v1",
        pair_index=index,
        original_context_indices=[oi, ni],
        partition=partition,
        source_cluster=off["source_index"] // 2,
        course=off["rows"][0]["course"],
        original_policy_hash=off["rows"][0]["original_policy_hash"],
        source_context_hashes=[off["report_hash"], on["report_hash"]],
        decision_frame=DECISION_FRAME,
        current_features=x.tolist(),
        current_features_hash=hash_bytes(x.tobytes()),
        observation_width=112,
        feature_role="SIM privileged diagnostic, not approved actor observation",
        future_outcomes_or_event_times_in_observation=False,
        nominal_target_epoch="current Foundation navigation before heading/IK/residual",
        action_names=list(ACTION_NAMES),
        outcomes_by_heading_then_action=outcomes,
        observed_safe_passing_actions_by_heading=[
            [i for i, o in enumerate(rows) if o["task_pass"] and o["safe"]] for rows in outcomes
        ],
        # Deadline failure is KNOWN even when complete ball quality is UNKNOWN.
        finite_horizon_task_pass_known_for_all26=True,
        incomplete_ball_quality_must_not_be_fabricated=True,
        original30_frame_parity_per_action=parities,
        repeated_zero_in_distinct_action_count=False,
        runtime_future_outcome_table_authorized=False,
        teacher_qualified=False,
        training_authorized=False,
        motor_policy_activation_authorized=False,
        hardware_authorized=False,
        promotion_authorized=False,
    )
    result["report_hash"] = hash_json(result)
    return result
