from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from rosclaw_soccer.media.free_kick_video import (
    _configure_offscreen_framebuffer,
    _qualify_g1_assets_headless,
    render_g1_free_kick_showcase_video,
)
from rosclaw_soccer.media.trajectory_render import (
    load_g1_ball_trajectory,
    sample_g1_ball_trajectory,
)


def test_free_kick_video_rejects_unknown_resolution(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="resolution must be"):
        render_g1_free_kick_showcase_video(
            evidence_path=tmp_path / "missing.json",
            asset_root=tmp_path / "assets",
            output_path=tmp_path / "video.mp4",
            source_checkout=tmp_path / "checkout",
            resolution="4k",
        )


def test_free_kick_video_expands_native_offscreen_framebuffer() -> None:
    model = SimpleNamespace(
        vis=SimpleNamespace(global_=SimpleNamespace(offwidth=640, offheight=480))
    )

    _configure_offscreen_framebuffer(model, width=1920, height=1080)

    assert model.vis.global_.offwidth == 1920
    assert model.vis.global_.offheight == 1080


def test_free_kick_video_selects_egl_before_asset_qualification(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    observed: list[str | None] = []

    def qualify(_asset_root: Path) -> object:
        observed.append(os.environ.get("MUJOCO_GL"))
        return object()

    monkeypatch.delenv("MUJOCO_GL", raising=False)
    monkeypatch.setattr("rosclaw_soccer.media.free_kick_video.qualify_g1_assets", qualify)

    qualification = _qualify_g1_assets_headless(tmp_path)

    assert qualification is not None
    assert observed == ["egl"]
    assert "MUJOCO_GL" not in os.environ


def test_free_kick_video_rejects_hardware_claim_before_asset_loading(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "pyproject.toml").write_text("[project]\nname='test'\n", encoding="utf-8")
    evidence = tmp_path / "evidence.json"
    evidence.write_text(
        json.dumps(
            {
                "passed": False,
                "strict_replay": True,
                "evidence_domain": "DEVELOPMENT_SHOWCASE",
                "activation_ceiling": "SIM_ONLY",
                "physics_authority": "CPU_MUJOCO",
                "hardware_command_sent": True,
                "claims": {
                    "real_hardware": False,
                    "rendered_pixels_used_for_scoring": False,
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "rosclaw_soccer.media.free_kick_video._qualify_g1_assets_headless",
        lambda _root: pytest.fail("asset qualification must not run"),
    )

    with pytest.raises(ValueError, match="hardware command"):
        render_g1_free_kick_showcase_video(
            evidence_path=evidence,
            asset_root=tmp_path / "assets",
            output_path=tmp_path / "outside.mp4",
            source_checkout=source,
            allow_rejected_candidate=True,
        )


def test_free_kick_video_rejects_raw_artifacts_inside_source(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "pyproject.toml").write_text("[project]\nname='test'\n", encoding="utf-8")

    with pytest.raises(ValueError, match="raw evidence must be outside"):
        render_g1_free_kick_showcase_video(
            evidence_path=source / "raw.json",
            asset_root=tmp_path / "assets",
            output_path=tmp_path / "outside.mp4",
            source_checkout=source,
        )


def test_render_trajectory_loader_and_interpolation_are_fail_closed(tmp_path: Path) -> None:
    path = tmp_path / "trajectory.npz"
    np.savez_compressed(
        path,
        time=np.asarray((0.0, 1.0), dtype=np.float64),
        pelvis_pose=np.asarray(
            ((0.0, 0.0, 0.8, 1.0, 0.0, 0.0, 0.0), (2.0, 0.0, 0.8, 1.0, 0.0, 0.0, 0.0))
        ),
        joint_position=np.asarray(((0.0,) * 29, (2.0,) * 29), dtype=np.float64),
        ball_pose=np.asarray(
            ((0.0, 0.0, 0.1, 1.0, 0.0, 0.0, 0.0), (4.0, 0.0, 0.1, 1.0, 0.0, 0.0, 0.0))
        ),
        scalar_metadata=np.asarray("bound-but-not-a-frame"),
    )

    trajectory = load_g1_ball_trajectory(path)
    _, pelvis, joints, ball = sample_g1_ball_trajectory(trajectory, 0.5)

    assert pelvis[0] == pytest.approx(1.0)
    assert np.allclose(joints, 1.0)
    assert ball[0] == pytest.approx(2.0)

    bad = tmp_path / "non-finite.npz"
    np.savez_compressed(
        bad,
        time=np.asarray((0.0, 1.0)),
        pelvis_pose=np.full((2, 7), np.nan),
        joint_position=np.zeros((2, 29)),
        ball_pose=np.zeros((2, 7)),
    )
    with pytest.raises(ValueError, match="non-finite"):
        load_g1_ball_trajectory(bad)
