"""Evidence-bound highlight reel for the low-to-aerial goalkeeper curriculum."""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import Any, BinaryIO, cast

import numpy as np

from rosclaw_soccer.media.shot_save_league_video import _load_trajectory, _write_frames
from rosclaw_soccer.media.shot_save_population_video import _atomic_json, _file_hash
from rosclaw_soccer.providers.g1.asset_qualification import qualify_g1_assets
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.training.goalkeeper_aerial_curriculum import (
    validate_goalkeeper_aerial_curriculum,
)
from rosclaw_soccer.world.field import G1TrainingGoalSpec

_CLIPS = (
    ("LOW:population-edgewide", "LOW FAR SAVE", "0x66E0FF"),
    ("MID:mid-left", "MID LEFT SAVE", "0x42E8A8"),
    ("HIGH_CENTER:high-center", "HIGH CENTER · GMT EXPERT", "0x42E8A8"),
    ("HIGH_INNER:high-inner-right", "HIGH INNER RIGHT", "0x42E8A8"),
    ("HIGH_CORNER:true-high-corner-left", "TRUE HIGH CORNER LEFT · 1.30 M+", "0xFFD166"),
    ("HIGH_CORNER:true-high-corner-right", "TRUE HIGH CORNER RIGHT · 1.30 M+", "0xFFD166"),
    ("HIGH_CORNER:frontier-corner-left", "FAST CORNER LEFT · 1.4 S", "0xFF8C42"),
    ("HIGH_CORNER:frontier-corner-right", "FAST CORNER RIGHT · 1.4 S", "0xFF8C42"),
)


def validate_aerial_curriculum_video_manifest(path: Path) -> dict[str, Any]:
    """Validate that pixels remain downstream of immutable scored trajectories."""

    manifest_path = path.expanduser().resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("aerial curriculum video manifest must be an object")
    expected = payload.pop("manifest_hash", None)
    if expected != hash_json(payload):
        raise ValueError("aerial curriculum video manifest integrity mismatch")
    video_value = payload.get("video_path")
    report_value = payload.get("report_path")
    if not isinstance(video_value, str) or not isinstance(report_value, str):
        raise ValueError("aerial curriculum video paths are invalid")
    video = Path(video_value).expanduser().resolve()
    report = Path(report_value).expanduser().resolve()
    numbers = tuple(
        payload.get(name) for name in ("fps", "width", "height", "frame_count", "duration_sec")
    )
    if (
        payload.get("schema_version") != "rosclaw_soccer.aerial_curriculum_video.v1"
        or not video.is_file()
        or payload.get("video_hash") != _file_hash(video)
        or not report.is_file()
        or payload.get("report_file_hash") != hash_bytes(report.read_bytes())
        or any(
            not isinstance(value, int | float)
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            or float(value) <= 0.0
            for value in numbers
        )
        or payload.get("visualization_only") is not True
        or payload.get("pixels_used_for_scoring") is not False
        or payload.get("promotion_eligible") is not False
        or payload.get("activation_ceiling") != "SIM_ONLY"
        or payload.get("hardware_command_sent") is not False
    ):
        raise ValueError("aerial curriculum video authority contract is invalid")
    payload["manifest_hash"] = expected
    return cast(dict[str, Any], payload)


def _ffmpeg_command(
    *, output: Path, width: int, height: int, fps: int, boundaries: tuple[float, ...]
) -> list[str]:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required for the aerial curriculum video")
    font = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    filters = [
        "drawbox=x=0:y=0:w=iw:h=116:color=black@0.64:t=fill",
        (
            f"drawtext=fontfile={font}:text='ROSClaw Goalkeeper Growth':"
            "fontcolor=white:fontsize=42:x=52:y=18"
        ),
        (
            f"drawtext=fontfile={font}:"
            "text='LOW TO FAST AERIAL CORNERS · SCORED CPU MUJOCO · SIM ONLY':"
            "fontcolor=0x66E0FF:fontsize=23:x=54:y=72"
        ),
    ]
    start = 0.0
    for (_, label, color), stop in zip(_CLIPS, boundaries, strict=True):
        filters.append(
            f"drawtext=fontfile={font}:text='{label}':fontcolor={color}:fontsize=32:"
            "x=(w-text_w)/2:y=h-86:box=1:boxcolor=black@0.58:boxborderw=14:"
            f"enable='between(t,{start:.3f},{stop:.3f})'"
        )
        start = stop
    return [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-vf",
        ",".join(filters),
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output),
    ]


def render_aerial_curriculum_video(
    *,
    report_path: Path,
    asset_root: Path,
    output_path: Path,
    source_checkout: Path,
    fps: int = 30,
    width: int = 1920,
    height: int = 1080,
) -> dict[str, Any]:
    """Render selected strict replays; never use pixels for scoring or promotion."""

    report_file = report_path.expanduser().resolve()
    output = output_path.expanduser().resolve()
    checkout = source_checkout.expanduser().resolve()
    if (
        output.exists()
        or output.suffix.lower() != ".mp4"
        or output == checkout
        or checkout in output.parents
        or not 20 <= fps <= 60
        or not 1280 <= width <= 3840
        or not 720 <= height <= 2160
    ):
        raise ValueError("aerial curriculum video output contract is invalid")
    report = validate_goalkeeper_aerial_curriculum(report_file)
    qualification = qualify_g1_assets(asset_root)
    qualification.require_eligible()
    if qualification.body_hash != report.get("body_hash"):
        raise ValueError("aerial curriculum video Body hash mismatch")
    rows = {f"LOW:{row['case_id']}": row for row in report["low_rows"]}
    rows.update({f"{row['band']}:{row['case_id']}": row for row in report["aerial_rows"]})
    if any(key not in rows or rows[key].get("passed") is not True for key, _, _ in _CLIPS):
        raise ValueError("aerial curriculum video requires passed scored clips")
    archives = report["trajectory_archives"]
    trajectories = {
        key: _load_trajectory(
            report_file.parent / archives[key]["path"],
            archives[key]["hash"],
        )
        for key, _, _ in _CLIPS
    }
    frame_groups = [
        np.arange(5.0, 10.40, 1.0 / fps, dtype=np.float64),
        *(np.arange(0.0, 2.05, 1.0 / fps, dtype=np.float64) for _ in range(len(_CLIPS) - 1)),
    ]
    durations = tuple(len(group) / fps for group in frame_groups)
    boundaries = tuple(float(value) for value in np.cumsum(durations))
    goals: dict[str, G1TrainingGoalSpec] = {}
    for key, _, _ in _CLIPS:
        row = rows[key]
        goals[key] = G1TrainingGoalSpec(
            plane_x_m=7.50,
            width_m=3.0,
            height_m=2.0,
            depth_m=1.2,
            target_y_m=float(row.get("target_y_m", 0.0)),
            target_z_m=float(row.get("target_z_m", 0.8)),
            precision_radius_m=0.10,
        )

    previous_gl = os.environ.get("MUJOCO_GL")
    os.environ.setdefault("MUJOCO_GL", "egl")
    import mujoco

    output.parent.mkdir(parents=True, exist_ok=True)
    process = subprocess.Popen(
        _ffmpeg_command(
            output=output,
            width=width,
            height=height,
            fps=fps,
            boundaries=boundaries,
        ),
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    if process.stdin is None:
        raise RuntimeError("aerial curriculum video ffmpeg pipe is unavailable")
    try:
        for (key, _, _), times in zip(_CLIPS, frame_groups, strict=True):
            _write_frames(
                mujoco=mujoco,
                asset_root=asset_root.expanduser().resolve(),
                trajectory=trajectories[key],
                goalkeeper_depth_m=0.25,
                goal=goals[key],
                times=times,
                width=width,
                height=height,
                stream=cast(BinaryIO, process.stdin),
            )
        process.stdin.close()
        stderr = process.stderr.read().decode(errors="replace") if process.stderr else ""
        if process.wait():
            raise RuntimeError(f"aerial curriculum video ffmpeg failed: {stderr[-3000:]}")
    except BaseException:
        process.stdin.close()
        process.kill()
        process.wait()
        raise
    finally:
        if previous_gl is None:
            os.environ.pop("MUJOCO_GL", None)
        else:
            os.environ["MUJOCO_GL"] = previous_gl
    manifest: dict[str, Any] = {
        "schema_version": "rosclaw_soccer.aerial_curriculum_video.v1",
        "video_path": str(output),
        "video_hash": _file_hash(output),
        "report_path": str(report_file),
        "report_hash": report["report_hash"],
        "report_file_hash": hash_bytes(report_file.read_bytes()),
        "selected_archives": {key: archives[key]["hash"] for key, _, _ in _CLIPS},
        "clips": [label for _, label, _ in _CLIPS],
        "fps": fps,
        "width": width,
        "height": height,
        "frame_count": int(sum(len(group) for group in frame_groups)),
        "duration_sec": float(sum(durations)),
        "visualization_only": True,
        "pixels_used_for_scoring": False,
        "promotion_eligible": False,
        "activation_ceiling": "SIM_ONLY",
        "hardware_command_sent": False,
        "goal_spec_template": asdict(G1TrainingGoalSpec(width_m=3.0, height_m=2.0)),
    }
    manifest["manifest_hash"] = hash_json(manifest)
    _atomic_json(output.with_suffix(".json"), manifest)
    return validate_aerial_curriculum_video_manifest(output.with_suffix(".json"))


__all__ = [
    "render_aerial_curriculum_video",
    "validate_aerial_curriculum_video_manifest",
]
