from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from rosclaw_soccer.media.three_role_save_portfolio_video import (
    _timeline,
    render_three_role_save_portfolio_video,
)


def test_save_portfolio_video_rejects_output_inside_checkout(tmp_path: Path) -> None:
    evidence = tmp_path / "evidence.json"
    evidence.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="output contract"):
        render_three_role_save_portfolio_video(
            evidence_path=evidence,
            asset_root=tmp_path,
            output_path=tmp_path / "checkout" / "video.mp4",
            source_checkout=tmp_path / "checkout",
        )


def test_save_portfolio_video_rejects_unqualified_evidence(tmp_path: Path) -> None:
    evidence = tmp_path / "evidence.json"
    evidence.write_text(json.dumps({"passed": False}), encoding="utf-8")
    with pytest.raises(ValueError, match="not render eligible"):
        render_three_role_save_portfolio_video(
            evidence_path=evidence,
            asset_root=tmp_path,
            output_path=tmp_path / "video.mp4",
            source_checkout=tmp_path / "checkout",
        )


def test_save_portfolio_titles_use_actual_case_count_and_contact_span() -> None:
    cases = {
        f"lane-{index}": {
            "lane": {"label": f"LANE {index}"},
            "replay": {
                "result": {
                    "pass_contact_time_sec": 1.0,
                    "shot_contact_time_sec": 2.0,
                    "goalkeeper_glove_contact_time_sec": 2.6,
                    "pass_delivery_error_m": 0.004,
                },
                "glove_contact_position_m": [7.0, 0.1 * index, 1.4],
                "incoming_speed_mps": 8.7,
            },
        }
        for index in range(3)
    }
    trajectories = {lane_id: {"time": np.array([0.0, 11.0])} for lane_id in cases}
    clips = _timeline(cases, trajectories, 30, 0.812)
    assert clips[0].label.startswith("3 SHOT LANES")
    assert clips[1].label.startswith("SAVE 1/3")
    assert clips[-1].label.startswith("3/3 STRICT")
    assert "0.812 m CONTACT SPAN" in clips[-1].label
