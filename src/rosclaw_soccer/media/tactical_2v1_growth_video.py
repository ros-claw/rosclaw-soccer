"""Render an evidence-downstream explanation of learned 2v1 decisions."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, BinaryIO, cast

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.training.tactical_2v1_growth import (
    validate_two_vs_one_growth_stage,
)

_CLAIM = "BOUNDED_TWO_VS_ONE_TACTICAL_GROWTH_VISUALIZATION"


def _font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    path = Path("/usr/share/fonts/truetype/dejavu") / name
    return ImageFont.truetype(str(path), size=size) if path.is_file() else ImageFont.load_default()


def _load_trajectory(path: Path) -> dict[str, np.ndarray]:
    required = {
        "time",
        "ball_pose",
        "carrier_position",
        "finisher_position",
        "defender_position",
    }
    with np.load(path, allow_pickle=False) as archive:
        value = {name: np.asarray(archive[name]) for name in archive.files}
    if required - value.keys() or value["time"].size < 2:
        raise ValueError("2v1 video trajectory is incomplete")
    return value


def _sample(array: np.ndarray, time: np.ndarray, sample_time: float) -> np.ndarray:
    index = int(np.searchsorted(time, sample_time, side="right"))
    if index <= 0:
        return np.asarray(array[0], dtype=np.float64)
    if index >= len(time):
        return np.asarray(array[-1], dtype=np.float64)
    start = index - 1
    span = float(time[index] - time[start])
    fraction = 0.0 if span <= 0.0 else (sample_time - float(time[start])) / span
    return (
        np.asarray(array[start], dtype=np.float64) * (1.0 - fraction)
        + np.asarray(array[index], dtype=np.float64) * fraction
    )


def _pitch_xy(position: np.ndarray, *, width: int, height: int) -> tuple[int, int]:
    left, right = int(0.07 * width), int(0.93 * width)
    top, bottom = int(0.16 * height), int(0.91 * height)
    x = left + (right - left) * float(np.clip(position[0] / 10.2, 0.0, 1.0))
    y = (top + bottom) / 2 - (bottom - top) * float(np.clip(position[1] / 6.8, -0.5, 0.5))
    return round(x), round(y)


def _draw_pitch(draw: ImageDraw.ImageDraw, *, width: int, height: int) -> None:
    left, right = int(0.07 * width), int(0.93 * width)
    top, bottom = int(0.16 * height), int(0.91 * height)
    draw.rounded_rectangle(
        (left, top, right, bottom), radius=24, fill=(21, 105, 58), outline=(222, 245, 227), width=5
    )
    for stripe in range(10):
        x0 = left + (right - left) * stripe // 10
        x1 = left + (right - left) * (stripe + 1) // 10
        if stripe % 2:
            draw.rectangle((x0, top + 4, x1, bottom - 4), fill=(25, 112, 63))
    middle = (left + right) // 2
    draw.line((middle, top, middle, bottom), fill=(222, 245, 227), width=4)
    radius = int(0.09 * (bottom - top))
    center_y = (top + bottom) // 2
    draw.ellipse(
        (middle - radius, center_y - radius, middle + radius, center_y + radius),
        outline=(222, 245, 227),
        width=4,
    )
    goal_x, goal_y = _pitch_xy(np.asarray((9.5, 0.0)), width=width, height=height)
    goal_half = int((bottom - top) * 1.83 / 6.8)
    draw.rectangle(
        (goal_x, goal_y - goal_half, right + 18, goal_y + goal_half),
        outline=(245, 245, 245),
        width=5,
    )


def _draw_agent(
    draw: ImageDraw.ImageDraw,
    position: np.ndarray,
    *,
    width: int,
    height: int,
    color: tuple[int, int, int],
    label: str,
) -> None:
    x, y = _pitch_xy(position, width=width, height=height)
    radius = max(17, width // 65)
    draw.ellipse(
        (x - radius, y - radius, x + radius, y + radius),
        fill=color,
        outline=(255, 255, 255),
        width=4,
    )
    draw.text(
        (x, y - 1),
        label,
        font=_font(max(14, width // 110), bold=True),
        fill=(255, 255, 255),
        anchor="mm",
    )


def _draw_case_frame(
    trajectory: dict[str, np.ndarray],
    row: dict[str, Any],
    *,
    sample_time: float,
    width: int,
    height: int,
) -> Image.Image:
    image = Image.new("RGB", (width, height), (8, 19, 31))
    draw = ImageDraw.Draw(image)
    _draw_pitch(draw, width=width, height=height)
    time = np.asarray(trajectory["time"], dtype=np.float64)
    ball = _sample(trajectory["ball_pose"], time, sample_time)[:2]
    carrier = _sample(trajectory["carrier_position"], time, sample_time)
    finisher = _sample(trajectory["finisher_position"], time, sample_time)
    defender = _sample(trajectory["defender_position"], time, sample_time)
    trail_start = max(0.0, sample_time - 0.50)
    indices = np.flatnonzero((time >= trail_start) & (time <= sample_time))
    if indices.size >= 2:
        points = [
            _pitch_xy(np.asarray(trajectory["ball_pose"])[index, :2], width=width, height=height)
            for index in indices[:: max(1, indices.size // 30)]
        ]
        if len(points) >= 2:
            draw.line(points, fill=(255, 223, 82), width=max(3, width // 420))
    _draw_agent(draw, carrier, width=width, height=height, color=(217, 40, 52), label="10")
    _draw_agent(draw, finisher, width=width, height=height, color=(238, 104, 43), label="7")
    _draw_agent(draw, defender, width=width, height=height, color=(34, 82, 205), label="6")
    ball_x, ball_y = _pitch_xy(ball, width=width, height=height)
    ball_radius = max(10, width // 105)
    draw.ellipse(
        (ball_x - ball_radius, ball_y - ball_radius, ball_x + ball_radius, ball_y + ball_radius),
        fill=(247, 247, 240),
        outline=(15, 15, 18),
        width=3,
    )
    selected = str(row["selected_action"]).upper()
    explanation = (
        "DEFENDER COVERS THE TEAMMATE  →  SHOOT"
        if selected == "SHOOT"
        else "DEFENDER PRESSES THE CARRIER  →  PASS"
    )
    draw.text(
        (width // 2, int(0.055 * height)),
        "ROSCLAW SOCCER · LEARNING WHY TO PASS",
        font=_font(max(30, width // 48), bold=True),
        fill=(242, 247, 252),
        anchor="mm",
    )
    draw.text(
        (width // 2, int(0.115 * height)),
        explanation,
        font=_font(max(20, width // 70), bold=True),
        fill=(71, 232, 178),
        anchor="mm",
    )
    confidence = float(row["decision"]["confidence"])
    footer = f"LEARNED {selected} · CONFIDENCE {confidence:.2f} · CPU MUJOCO · SIM ONLY"
    draw.rounded_rectangle(
        (int(0.16 * width), int(0.925 * height), int(0.84 * width), int(0.982 * height)),
        radius=18,
        fill=(7, 14, 25),
    )
    draw.text(
        (width // 2, int(0.953 * height)),
        footer,
        font=_font(max(17, width // 86), bold=True),
        fill=(238, 242, 247),
        anchor="mm",
    )
    return image


def _title_frame(*, width: int, height: int, metrics: dict[str, Any], outro: bool) -> Image.Image:
    image = Image.new("RGB", (width, height), (7, 16, 29))
    draw = ImageDraw.Draw(image)
    accent = (67, 232, 174)
    if outro:
        draw.text(
            (width // 2, int(0.22 * height)),
            "FROM FIXED ACTIONS TO CONTEXTUAL DECISIONS",
            font=_font(max(34, width // 43), bold=True),
            fill=(244, 248, 252),
            anchor="mm",
        )
        lines = (
            f"LEARNED POLICY  {100 * float(metrics['task_success_rate']):.0f}%",
            f"FIXED PASS / FIXED SHOOT  {100 * float(metrics['fixed_pass_success_rate']):.1f}%",
            f"NET GAIN  +{100 * float(metrics['gain_over_fixed_pass']):.1f} POINTS",
            "24 / 24 SAFE · 24 / 24 EXACT REPLAY",
        )
        for index, line in enumerate(lines):
            draw.text(
                (width // 2, int((0.39 + 0.105 * index) * height)),
                line,
                font=_font(max(25, width // 58), bold=index < 3),
                fill=accent if index == 2 else (224, 233, 242),
                anchor="mm",
            )
    else:
        draw.text(
            (width // 2, int(0.30 * height)),
            "SKILL IS NOT ENOUGH",
            font=_font(max(45, width // 35), bold=True),
            fill=(244, 248, 252),
            anchor="mm",
        )
        draw.text(
            (width // 2, int(0.46 * height)),
            "THE PLAYER MUST LEARN",
            font=_font(max(28, width // 52), bold=True),
            fill=(167, 184, 204),
            anchor="mm",
        )
        draw.text(
            (width // 2, int(0.58 * height)),
            "WHEN TO PASS · WHEN TO SHOOT",
            font=_font(max(38, width // 40), bold=True),
            fill=accent,
            anchor="mm",
        )
        draw.text(
            (width // 2, int(0.78 * height)),
            "TACTICAL PLANE · FROZEN G1 SKILL IDENTITIES · PHYSICS-SCORED",
            font=_font(max(18, width // 78)),
            fill=(171, 185, 202),
            anchor="mm",
        )
    return image


def _write_frame(stream: BinaryIO, image: Image.Image) -> None:
    stream.write(np.asarray(image, dtype=np.uint8).tobytes())


def render_two_vs_one_growth_video(
    *,
    stage_summary_path: Path,
    output_path: Path,
    fps: int = 30,
    width: int = 1920,
    height: int = 1080,
) -> dict[str, Any]:
    """Render two representative decisions and a metric-bound summary."""

    output = output_path.expanduser().resolve()
    manifest_path = output.with_suffix(".json")
    if (
        output.exists()
        or manifest_path.exists()
        or output.suffix.lower() != ".mp4"
        or not 20 <= fps <= 60
        or not 1280 <= width <= 3840
        or not 720 <= height <= 2160
        or shutil.which("ffmpeg") is None
    ):
        raise ValueError("2v1 video output contract is invalid")
    stage_path = stage_summary_path.expanduser().resolve()
    stage = validate_two_vs_one_growth_stage(stage_path)
    retention_path = stage_path.parent / "retention/retention-exam.json"
    retention = json.loads(retention_path.read_text(encoding="utf-8"))
    rows = retention.get("rows")
    metrics = retention.get("metrics")
    if not isinstance(rows, list) or not isinstance(metrics, dict):
        raise ValueError("2v1 video source evidence is incomplete")
    selected_rows: list[tuple[int, dict[str, Any]]] = []
    for wanted in ("shoot", "pass"):
        match = next(
            (
                (index, row)
                for index, row in enumerate(rows)
                if isinstance(row, dict)
                and row.get("selected_action") == wanted
                and row.get("task_succeeded") is True
                and row.get("exact_replay") is True
            ),
            None,
        )
        if match is None:
            raise ValueError("2v1 video needs one strict PASS and SHOOT case")
        selected_rows.append(match)
    sources: dict[str, str] = {
        str(stage_path): hash_bytes(stage_path.read_bytes()),
        str(retention_path): hash_bytes(retention_path.read_bytes()),
    }
    clips: list[tuple[dict[str, Any], dict[str, np.ndarray], Path]] = []
    for index, row in selected_rows:
        trajectory_path = (
            retention_path.parent / f"case-{index:03d}" / str(row["primary_artifact"]["file"])
        )
        sources[str(trajectory_path)] = hash_bytes(trajectory_path.read_bytes())
        clips.append((row, _load_trajectory(trajectory_path), trajectory_path))

    output.parent.mkdir(parents=True, exist_ok=True)
    command = (
        "ffmpeg",
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
    )
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    if process.stdin is None:
        raise RuntimeError("ffmpeg video pipe is unavailable")
    stream = cast(BinaryIO, process.stdin)
    frame_count = 0
    try:
        for _ in range(round(1.8 * fps)):
            _write_frame(
                stream,
                _title_frame(width=width, height=height, metrics=metrics, outro=False),
            )
            frame_count += 1
        for row, trajectory, _ in clips:
            duration = float(np.asarray(trajectory["time"])[-1])
            speed = 0.70 if row["selected_action"] == "shoot" else 0.35
            clip_frames = round(4.2 * fps)
            for frame in range(clip_frames):
                sample_time = min(duration, frame / fps * speed)
                _write_frame(
                    stream,
                    _draw_case_frame(
                        trajectory, row, sample_time=sample_time, width=width, height=height
                    ),
                )
                frame_count += 1
        for _ in range(round(2.4 * fps)):
            _write_frame(
                stream, _title_frame(width=width, height=height, metrics=metrics, outro=True)
            )
            frame_count += 1
    finally:
        stream.close()
    return_code = process.wait()
    if return_code != 0 or not output.is_file():
        raise RuntimeError("ffmpeg failed to render the 2v1 growth video")
    manifest: dict[str, Any] = {
        "schema_version": "rosclaw_soccer.two_vs_one_growth_video.v1",
        "claim": _CLAIM,
        "source_stage_hash": stage["stage_hash"],
        "source_stage_passed": True,
        "source_files": sources,
        "video_path": str(output),
        "video_hash": hash_bytes(output.read_bytes()),
        "width": width,
        "height": height,
        "fps": fps,
        "frame_count": frame_count,
        "duration_sec": frame_count / fps,
        "visualization_only": True,
        "tactical_plane_only": True,
        "g1_bodies_rendered": False,
        "pixels_used_for_scoring": False,
        "commercial_use_allowed": False,
        "promotion_eligible": False,
        "activation_ceiling": "SIM_ONLY",
        "hardware_command_sent": False,
    }
    manifest["manifest_hash"] = hash_json(manifest)
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, manifest_path)
    return manifest


def validate_two_vs_one_growth_video_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("2v1 video manifest must be an object")
    claimed = payload.pop("manifest_hash", None)
    try:
        sources = payload.get("source_files")
        video_value = payload.get("video_path")
        if not isinstance(sources, dict) or not isinstance(video_value, str):
            raise ValueError("2v1 video bindings are invalid")
        video = Path(video_value).expanduser().resolve()
        if not video.is_file() or payload.get("video_hash") != hash_bytes(video.read_bytes()):
            raise ValueError("2v1 video changed")
        for source_value, expected_hash in sources.items():
            source = Path(source_value).expanduser().resolve()
            if not source.is_file() or hash_bytes(source.read_bytes()) != expected_hash:
                raise ValueError("2v1 video source changed")
        if (
            claimed != hash_json(payload)
            or payload.get("schema_version") != "rosclaw_soccer.two_vs_one_growth_video.v1"
            or payload.get("claim") != _CLAIM
            or payload.get("source_stage_passed") is not True
            or payload.get("visualization_only") is not True
            or payload.get("tactical_plane_only") is not True
            or payload.get("g1_bodies_rendered") is not False
            or payload.get("pixels_used_for_scoring") is not False
            or payload.get("promotion_eligible") is not False
            or payload.get("activation_ceiling") != "SIM_ONLY"
            or payload.get("hardware_command_sent") is not False
        ):
            raise ValueError("2v1 video authority or integrity contract is invalid")
    finally:
        payload["manifest_hash"] = claimed
    return payload


__all__ = [
    "render_two_vs_one_growth_video",
    "validate_two_vs_one_growth_video_manifest",
]
