from __future__ import annotations

import inspect

import numpy as np

from rosclaw_soccer.media.contextual_strike_expert_video import (
    _segments,
    render_contextual_strike_expert_video,
    validate_contextual_strike_expert_video_manifest,
)


def test_contextual_video_is_evidence_downstream() -> None:
    parameters = inspect.signature(render_contextual_strike_expert_video).parameters
    for name in ("exam_path", "asset_root", "output_path"):
        assert parameters[name].kind is inspect.Parameter.KEYWORD_ONLY
    assert callable(validate_contextual_strike_expert_video_manifest)


def test_contextual_video_discloses_holdout_failure_and_rejected_promotion() -> None:
    names = (
        "baseline0700",
        "training0700",
        "training0800",
        "holdout07125",
        "holdout07375",
    )
    trajectories = {name: {"time": np.asarray((0.0, 8.6))} for name in names}
    reports = {name: {"assessment": {"events": {"strike_time_sec": 7.5}}} for name in names}

    segments = _segments(trajectories=trajectories, reports=reports, fps=30)
    labels = " ".join(clip.title for _, clips in segments for clip in clips)

    assert "DAGGER REPAIR" in labels
    assert "UNSEEN MIDPOINT y=0.7125 m · PASS" in labels
    assert "FAILURE DISCLOSED" in labels
    assert "PROMOTION REJECTED" in labels
