from __future__ import annotations

import json
from pathlib import Path

import pytest

from rosclaw_soccer.media.three_role_save_portfolio_video import (
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
