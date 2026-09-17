"""Audit measured assisted restarts separately from player achievements."""

from typing import Any

import numpy as np

from rosclaw_soccer.training.returned_ball_learning import returned_ball_segments


def return_turnaround_metrics(time: np.ndarray, event_code: np.ndarray) -> dict[str, Any]:
    """Use observed exit/release/re-entry times; never infer an unseen return."""
    # Reuse lifecycle validation, including a failed outside release followed
    # by another exit. Only two small columns are copied for that validation.
    returned_ball_segments(dict(time=time, training_return_event_code=event_code))
    rows: list[dict[str, Any]] = []
    pending: dict[str, Any] | None = None
    for frame in np.flatnonzero(event_code):
        now = float(time[frame])
        code = int(event_code[frame])
        if code == 1:
            pending = dict(exit_sec=now, release_sec=None, reentry_sec=None)
            rows.append(pending)
        elif code == 2:
            assert pending is not None
            pending["release_sec"] = now
        elif code == 3:
            assert pending is not None
            pending["reentry_sec"] = now
    for row in rows:
        row["exit_to_release_sec"] = (
            None if row["release_sec"] is None else row["release_sec"] - row["exit_sec"]
        )
        row["exit_to_reentry_sec"] = (
            None if row["reentry_sec"] is None else row["reentry_sec"] - row["exit_sec"]
        )
    return dict(
        observed_exits=len(rows),
        external_releases=sum(r["release_sec"] is not None for r in rows),
        observed_reentries=sum(r["reentry_sec"] is not None for r in rows),
        unreleased_exits=sum(r["release_sec"] is None for r in rows),
        rows=rows,
        external_returns_are_player_passes=False,
        match_qualified=False,
    )
