"""Evidence-downstream highlight reel for the S79 population exam."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any, BinaryIO, cast

import numpy as np

from rosclaw_soccer.media.shot_save_league_video import _load_trajectory, _write_frames
from rosclaw_soccer.providers.g1.asset_qualification import qualify_g1_assets
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.training.shot_save_population import validate_shot_save_population_exam
from rosclaw_soccer.world.field import G1TrainingGoalSpec


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def validate_shot_save_population_video_manifest(path: Path) -> dict[str, Any]:
    manifest_path = path.expanduser().resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("population video manifest must be a JSON object")
    expected = payload.pop("manifest_hash", None)
    if expected != hash_json(payload):
        raise ValueError("population video manifest integrity mismatch")
    video_value = payload.get("video_path")
    report_value = payload.get("report_path")
    if not isinstance(video_value, str) or not isinstance(report_value, str):
        raise ValueError("population video content paths are invalid")
    video = Path(video_value).expanduser().resolve()
    report = Path(report_value).expanduser().resolve()
    if (
        not video.is_file()
        or payload.get("video_hash") != _file_hash(video)
        or not report.is_file()
        or payload.get("report_file_hash") != hash_bytes(report.read_bytes())
        or payload.get("schema_version") != "rosclaw_soccer.shot_save_population_video.v1"
        or payload.get("visualization_only") is not True
        or payload.get("pixels_used_for_scoring") is not False
        or payload.get("promotion_eligible") is not False
        or payload.get("activation_ceiling") != "SIM_ONLY"
        or payload.get("hardware_command_sent") is not False
    ):
        raise ValueError("population video authority or content binding is invalid")
    payload["manifest_hash"] = expected
    return cast(dict[str, Any], payload)


def _ffmpeg_command(
    *, output: Path, width: int, height: int, fps: int, boundaries: tuple[float, ...]
) -> list[str]:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required for the population video")
    font = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    labels = (
        (0.0, boundaries[0], "PARENT KEEPER · FAR SHOT · GOAL", "0xFFB000"),
        (boundaries[0], boundaries[1], "NEW KEEPER · SAME SHOT · SAVE", "0x42E8A8"),
        (boundaries[1], boundaries[2], "NEW KEEPER · CENTER SAVE", "0x66E0FF"),
        (boundaries[2], boundaries[3], "NEW KEEPER · POST-CORNER SAVE", "0x66E0FF"),
    )
    filters = [
        "drawbox=x=0:y=0:w=iw:h=116:color=black@0.64:t=fill",
        (
            f"drawtext=fontfile={font}:text='ROSClaw Population Growth':"
            "fontcolor=white:fontsize=42:x=52:y=18"
        ),
        (
            f"drawtext=fontfile={font}:"
            "text='8-SHOT PAIRED EXAM · SCORED PHYSICS · SIM ONLY':"
            "fontcolor=0x66E0FF:fontsize=23:x=54:y=72"
        ),
    ]
    for start, stop, label, color in labels:
        filters.append(
            f"drawtext=fontfile={font}:text='{label}':fontcolor={color}:fontsize=32:"
            "x=(w-text_w)/2:y=h-86:box=1:boxcolor=black@0.58:boxborderw=14:"
            f"enable='between(t,{start:.3f},{stop:.3f})'"
        )
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


def render_shot_save_population_video(
    *,
    report_path: Path,
    asset_root: Path,
    output_path: Path,
    source_checkout: Path,
    fps: int = 30,
    width: int = 1920,
    height: int = 1080,
) -> dict[str, Any]:
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
        raise ValueError("population video output contract is invalid")
    report = validate_shot_save_population_exam(report_file)
    previous_gl = os.environ.get("MUJOCO_GL")
    os.environ.setdefault("MUJOCO_GL", "egl")
    import mujoco

    qualification = qualify_g1_assets(asset_root)
    qualification.require_eligible()
    if qualification.body_hash != report.get("body_hash"):
        raise ValueError("population video Body hash does not match evidence")
    choices = (
        ("parent", "population-far"),
        ("candidate", "population-far"),
        ("candidate", "population-center"),
        ("candidate", "population-edgewide"),
    )
    cells = {
        ("parent", cell["striker_policy_id"]): cell for cell in report["parent_cells"]
    }
    cells.update(
        {
            ("candidate", cell["striker_policy_id"]): cell
            for cell in report["candidate_cells"]
        }
    )
    if cells[choices[0]]["result"]["goal_crossed"] is not True or any(
        cells[choice]["keeper_exam_passed"] is not True for choice in choices[1:]
    ):
        raise ValueError("population video selected outcomes do not match the scored exam")
    shots = {item["policy_id"]: item for item in report["config"]["shots"]}
    keepers = {
        "parent": report["config"]["parent_goalkeeper"],
        "candidate": report["config"]["candidate_goalkeeper"],
    }
    archives = report["trajectory_archives"]
    trajectories = {
        choice: _load_trajectory(
            report_file.parent / archives[f"{choice[0]}:{choice[1]}"]["path"],
            archives[f"{choice[0]}:{choice[1]}"]["hash"],
        )
        for choice in choices
    }
    times = np.arange(4.60, 10.30, 1.0 / fps, dtype=np.float64)
    durations = tuple(len(times) / fps for _ in choices)
    boundaries = tuple(float(value) for value in np.cumsum(durations))
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
        raise RuntimeError("population video ffmpeg pipe is unavailable")
    try:
        for role, shot_id in choices:
            shot = shots[shot_id]
            goal = G1TrainingGoalSpec(
                plane_x_m=7.50,
                width_m=3.0,
                height_m=2.0,
                depth_m=1.2,
                target_y_m=float(shot["physical_target_y_m"]),
                target_z_m=float(shot["physical_target_z_m"]),
                precision_radius_m=0.10,
            )
            _write_frames(
                mujoco=mujoco,
                asset_root=asset_root.expanduser().resolve(),
                trajectory=trajectories[(role, shot_id)],
                goalkeeper_depth_m=float(keepers[role]["depth_from_goal_line_m"]),
                goal=goal,
                times=times,
                width=width,
                height=height,
                stream=cast(BinaryIO, process.stdin),
            )
        process.stdin.close()
        stderr = process.stderr.read().decode(errors="replace") if process.stderr else ""
        if process.wait():
            raise RuntimeError(f"population video ffmpeg failed: {stderr[-3000:]}")
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
        "schema_version": "rosclaw_soccer.shot_save_population_video.v1",
        "video_path": str(output),
        "video_hash": _file_hash(output),
        "report_path": str(report_file),
        "report_hash": report["report_hash"],
        "report_file_hash": hash_bytes(report_file.read_bytes()),
        "selected_archives": {
            f"{role}:{shot}": archives[f"{role}:{shot}"]["hash"] for role, shot in choices
        },
        "fps": fps,
        "width": width,
        "height": height,
        "frame_count": int(len(times) * len(choices)),
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
    return validate_shot_save_population_video_manifest(output.with_suffix(".json"))


__all__ = [
    "render_shot_save_population_video",
    "validate_shot_save_population_video_manifest",
]
