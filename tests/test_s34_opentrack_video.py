from __future__ import annotations

from pathlib import Path

import pytest

from rosclaw_soccer.evidence.opentrack_exam import OpenTrackEpisodeSpec
from rosclaw_soccer.evidence.opentrack_video import render_opentrack_residual_videos


def _episode() -> OpenTrackEpisodeSpec:
    return OpenTrackEpisodeSpec(
        episode_id="left-jump",
        suite_id="acquisition",
        dataset_id="lafan1",
        motion_id="s34m_leftjump",
        source_hash="sha256:" + "0" * 64,
        license_id="CC-BY-NC-ND-4.0",
        start_frame=0,
        max_steps=100,
        critical=False,
    )


def test_video_renderer_fails_before_runtime_import_when_inputs_are_missing(
    tmp_path: Path,
) -> None:
    with pytest.raises(FileNotFoundError, match="inputs do not exist"):
        render_opentrack_residual_videos(
            opentrack_root=tmp_path / "missing",
            candidate_policy_path=tmp_path / "candidate.onnx",
            parent_policy_path=tmp_path / "parent.onnx",
            source_config_path=tmp_path / "config.json",
            episodes=(_episode(),),
            residual_scale=0.25,
            output_dir=tmp_path / "evidence",
        )


def test_video_renderer_rejects_evidence_inside_source_checkout(tmp_path: Path) -> None:
    root = tmp_path / "opentrack"
    root.mkdir()
    inputs = [root / "candidate.onnx", root / "parent.onnx", root / "config.json"]
    for path in inputs:
        path.write_text("sealed", encoding="utf-8")

    with pytest.raises(ValueError, match="outside the source checkout"):
        render_opentrack_residual_videos(
            opentrack_root=root,
            candidate_policy_path=inputs[0],
            parent_policy_path=inputs[1],
            source_config_path=inputs[2],
            episodes=(_episode(),),
            residual_scale=0.25,
            output_dir=root / "evidence",
        )
