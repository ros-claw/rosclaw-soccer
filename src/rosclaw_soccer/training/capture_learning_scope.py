"""Prepare actual capture-phase learning without relabeling sampled actions.

This is an eligibility audit, not an on-policy provenance check or a football
success certificate. Callers must still reconstruct sampled actions and verify
the frozen policy, physical configuration and collection evidence.
"""

from collections.abc import Mapping
from typing import Any

import numpy as np
from numpy.typing import NDArray

from rosclaw_soccer.training.returned_ball_learning import returned_ball_segments


def capture_learning_segments(
    trace: Mapping[str, Any], *, exploration_mask: NDArray[np.bool_]
) -> tuple[list[dict[str, NDArray[Any]]], dict[str, Any]]:
    """Select measured live-foundation capture windows from a returned-ball course.

    The exploration mask must be recorded at action generation, not synthesized
    from later success. It remains unchanged even where a motor guard blocked
    physical application. PPO intersects exploration, physical activation and
    task eligibility itself. Tactical PASS/RECEIVE labels do not define the
    executed control phase. No reward crosses a coach reset.
    """
    if any(key in trace for key in ("residual_learning_mask", "residual_exploration_mask")):
        raise ValueError("learning or exploration labels already exist; refusing to overwrite")
    if "training_return_event_code" not in trace:
        raise ValueError("observed returned-live lifecycle required")
    try:
        time = np.asarray(trace["time"])
        context = np.asarray(trace["post_receive_capture_context"])
        foundation = np.asarray(trace["capture_live_foundation_active"])
        active = np.asarray(trace["residual_active"])
    except KeyError as exc:
        raise ValueError("explicit measured capture, foundation and active masks required") from exc
    exploration = np.asarray(exploration_mask)
    if (
        time.ndim != 1
        or context.ndim != 2
        or context.shape[0] != len(time)
        or context.shape[1] == 0
        or any(
            mask.dtype != np.bool_ or mask.shape != context.shape
            for mask in (context, foundation, active, exploration)
        )
    ):
        raise ValueError("aligned boolean frame-by-player capture masks required")
    if np.any(foundation & ~context):
        raise ValueError("live foundation activation must be a subset of capture context")
    if np.any(exploration & ~context):
        raise ValueError("capture-only sampling explored outside measured capture context")
    if np.any(exploration & ~foundation):
        raise ValueError("capture training requires the live foundation on explored frames")
    # Lifecycle validation precedes masking: malformed events are never repaired.
    prepared = dict(trace)
    prepared["residual_exploration_mask"] = exploration.copy()
    prepared["residual_learning_mask"] = (context & foundation).copy()
    segments = returned_ball_segments(prepared)
    live_exploration = sum(int(p["residual_exploration_mask"].sum()) for p in segments)
    if live_exploration != int(exploration.sum()):
        raise ValueError("exploration must be confined to observed returned-live epochs")
    eligible = np.zeros(context.shape[1], dtype=np.int64)
    explored = np.zeros_like(eligible)
    for part in segments:
        mask = (
            part["residual_exploration_mask"]
            & part["residual_learning_mask"]
            & part["residual_active"]
        )
        eligible += mask.sum(axis=0)
        explored += part["residual_exploration_mask"].sum(axis=0)
    return segments, {
        "schema": "soccer.capture_learning_scope.v1",
        "raw_frames": len(time),
        "live_frames": sum(len(p["time"]) for p in segments),
        "live_segments": len(segments),
        "exploration_samples_by_column": explored.tolist(),
        "eligible_samples_by_column": eligible.tolist(),
        "eligible_samples": int(eligible.sum()),
        "activation_ceiling": "SIM_ONLY",
        "promotion_eligible": False,
    }
