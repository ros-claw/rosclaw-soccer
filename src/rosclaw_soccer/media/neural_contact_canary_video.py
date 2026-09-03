"""Evidence-downstream 1080p reel for the S130 neural contact canary."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, BinaryIO, cast

import numpy as np

from rosclaw_soccer.media.three_player_video import (
    ThreePlayerVideoClip,
    _configure_offscreen_framebuffer,
    _Frame,
    _probe_video,
    _segment,
    _write_frames,
)
from rosclaw_soccer.media.trajectory_render import escape_filtergraph_option
from rosclaw_soccer.providers.g1.asset_qualification import qualify_g1_assets
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.skills.team.development_evidence import (
    three_role_development_kwargs,
    three_role_goal_spec,
)
from rosclaw_soccer.skills.team.shared_world import G1GoalkeeperConfig
from rosclaw_soccer.training.neural_contact_canary import (
    validate_neural_contact_canary,
)
from rosclaw_soccer.world.field import build_g1_three_player_stadium_model


def render_neural_contact_canary_video(
    *,
    canary_report_path: Path,
    asset_root: Path,
    output_path: Path,
    source_checkout: Path,
    fps: int = 30,
    width: int = 1920,
    height: int = 1080,
) -> dict[str, Any]:
    report_path = canary_report_path.expanduser().resolve()
    report = validate_neural_contact_canary(report_path)
    output = output_path.expanduser().resolve()
    checkout = source_checkout.expanduser().resolve()
    manifest_path = output.with_suffix(".json")
    if (
        output.exists()
        or manifest_path.exists()
        or output.suffix.lower() != ".mp4"
        or output == checkout
        or checkout in output.parents
        or not 20 <= fps <= 60
        or not 1280 <= width <= 3840
        or not 720 <= height <= 2160
    ):
        raise ValueError("neural contact video output contract is invalid")
    primary = next(run for run in report["runs"] if run["label"] == "candidate-primary")
    artifact = primary["trajectory"]
    trajectory_path = report_path.parent / artifact["file"]
    if hash_bytes(trajectory_path.read_bytes()) != artifact["file_hash"]:
        raise ValueError("neural contact video trajectory binding changed")
    with np.load(trajectory_path, allow_pickle=False) as archive:
        trajectory = {name: np.asarray(archive[name]) for name in archive.files}
    request = json.loads((report_path.parent / "request.json").read_text(encoding="utf-8"))
    qualification = qualify_g1_assets(asset_root)
    qualification.require_eligible()
    if qualification.body_hash != request["body_hash"]:
        raise ValueError("neural contact video Body identity changed")
    result = primary["result"]
    timelines, clips = _timeline(result, trajectory, fps)
    goal = three_role_goal_spec()
    runtime_kwargs = three_role_development_kwargs()
    keeper = runtime_kwargs.get("goalkeeper_config")
    if not isinstance(keeper, G1GoalkeeperConfig):
        raise ValueError("neural contact video goalkeeper contract is missing")
    bundle = SimpleNamespace(
        request={
            "physical_scoring_target_m": [goal.plane_x_m, goal.target_y_m, goal.target_z_m],
            "goal_spec": asdict(goal),
        },
        report={"result": result},
        trajectory=trajectory,
    )
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if ffmpeg is None or ffprobe is None:
        raise RuntimeError("ffmpeg and ffprobe are required for neural contact video")
    previous_gl = os.environ.get("MUJOCO_GL")
    os.environ.setdefault("MUJOCO_GL", "egl")
    try:
        import mujoco

        context = request["context"]
        passer_origin = tuple(float(value) for value in context["passer_origin_m"])
        model = build_g1_three_player_stadium_model(
            asset_root.expanduser().resolve(),
            passer_origin_m=cast(tuple[float, float, float], passer_origin),
            goalkeeper_origin_m=(goal.plane_x_m - keeper.depth_from_goal_line_m, 0.0, 0.0),
            spec=goal,
        )
        _configure_offscreen_framebuffer(model, width=width, height=height)
        data = mujoco.MjData(model)
        renderer = mujoco.Renderer(model, height=height, width=width)
        output.parent.mkdir(parents=True, exist_ok=True)
        try:
            with tempfile.TemporaryDirectory(prefix="rosclaw-neural-contact-video-") as temp:
                labels = _write_labels(Path(temp), result)
                process = subprocess.Popen(
                    _ffmpeg_command(
                        ffmpeg=ffmpeg,
                        output=output,
                        fps=fps,
                        width=width,
                        height=height,
                        labels=labels,
                        clips=clips,
                    ),
                    stdin=subprocess.PIPE,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                )
                if process.stdin is None:
                    raise RuntimeError("neural contact ffmpeg pipe is unavailable")
                try:
                    _write_frames(
                        mujoco=mujoco,
                        model=model,
                        data=data,
                        renderer=renderer,
                        bundle=cast(Any, bundle),
                        timelines=timelines,
                        stream=cast(BinaryIO, process.stdin),
                    )
                except BaseException:
                    process.stdin.close()
                    process.kill()
                    process.wait()
                    raise
                process.stdin.close()
                stderr = process.stderr.read().decode(errors="replace") if process.stderr else ""
                if process.wait():
                    raise RuntimeError(f"neural contact ffmpeg failed: {stderr[-3000:]}")
        finally:
            renderer.close()
    finally:
        if previous_gl is None:
            os.environ.pop("MUJOCO_GL", None)
        else:
            os.environ["MUJOCO_GL"] = previous_gl
    probe = _probe_video(ffprobe, output)
    frame_count = sum(clip.frame_count for clip in clips)
    if (
        probe["width"] != width
        or probe["height"] != height
        or probe["fps"] != fps
        or abs(probe["frame_count"] - frame_count) > 1
    ):
        raise RuntimeError("neural contact encoded video contract changed")
    manifest: dict[str, Any] = {
        "schema_version": "rosclaw_soccer.neural_contact_canary_video.v1",
        "video_path": str(output),
        "video_hash": hash_bytes(output.read_bytes()),
        "source_canary_path": str(report_path),
        "source_canary_hash": hash_bytes(report_path.read_bytes()),
        "source_canary_report_hash": report["report_hash"],
        "trajectory_hash": artifact["file_hash"],
        "actor_hash": report["actor_hash"],
        "fps": fps,
        "width": width,
        "height": height,
        "frame_count": frame_count,
        "duration_sec": frame_count / fps,
        "clips": [asdict(clip) for clip in clips],
        "visualization_only": True,
        "pixels_used_for_scoring": False,
        "promotion_eligible": False,
        "activation_ceiling": "SIM_ONLY",
        "hardware_command_sent": False,
    }
    manifest["manifest_hash"] = hash_json(manifest)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def validate_neural_contact_canary_video(path: Path) -> dict[str, Any]:
    manifest_path = path.expanduser().resolve()
    payload = cast(dict[str, Any], json.loads(manifest_path.read_text(encoding="utf-8")))
    claimed = payload.pop("manifest_hash", None)
    if claimed != hash_json(payload):
        raise ValueError("neural contact video manifest changed")
    video = Path(payload["video_path"])
    if (
        hash_bytes(video.read_bytes()) != payload["video_hash"]
        or payload.get("visualization_only") is not True
        or payload.get("pixels_used_for_scoring") is not False
    ):
        raise ValueError("neural contact video binding is invalid")
    payload["manifest_hash"] = claimed
    return payload


def _timeline(
    result: dict[str, Any], trajectory: dict[str, np.ndarray], fps: int
) -> tuple[tuple[tuple[_Frame, ...], ...], tuple[ThreePlayerVideoClip, ...]]:
    start = float(trajectory["time"][0])
    end = float(trajectory["time"][-1])
    pass_time = float(result["pass_contact_time_sec"])
    shot_time = float(result["shot_contact_time_sec"])
    timelines = (
        tuple(_Frame(start, "wide") for _ in range(round(1.3 * fps))),
        (
            *_segment(max(start, 4.4), pass_time + 0.35, 1.0, "pass", fps),
            *_segment(pass_time + 0.35, shot_time - 0.45, 1.0, "roll", fps),
            *_segment(shot_time - 0.45, min(end, shot_time + 1.35), 1.0, "goal_field", fps),
        ),
        _segment(shot_time - 0.70, min(end, shot_time + 1.35), 0.32, "goal_front", fps),
        _segment(shot_time + 0.35, end, 0.72, "recovery_shooter", fps),
        tuple(_Frame(end, "wide") for _ in range(round(1.8 * fps))),
    )
    specs = (
        ("01-intro", "NEURAL CONTACT MUSCLE MEMORY", "VERIFIED_POSE_HOLD"),
        ("02-chain", "PASS → RIGHT-FOOT GOAL", "STRICT_PHYSICS_REPLAY"),
        ("03-contact", "29-DOF NEURAL TORQUE", "SLOW_MOTION_REPLAY"),
        ("04-recovery", "STABLE RECOVERY", "INTERPOLATED_REPLAY"),
        ("05-score", "SEALED CAUSAL CANARY PASS", "VERIFIED_FINAL_POSE_HOLD"),
    )
    clips = tuple(
        ThreePlayerVideoClip(
            clip_id=clip_id,
            title=title,
            frame_count=len(timeline),
            duration_sec=len(timeline) / fps,
            playback_kind=kind,
        )
        for timeline, (clip_id, title, kind) in zip(timelines, specs, strict=True)
    )
    return timelines, clips


def _write_labels(root: Path, result: dict[str, Any]) -> tuple[Path, ...]:
    values = (
        "S130 · TWO-LAYER PROPRIOCEPTIVE CEREBELLUM · 29 JOINT TORQUES",
        f"MEASURED PASS → RIGHT FOOT → GOAL · {result['shot_peak_ball_speed_mps']:.2f} m/s",
        "NEURAL ACTOR ONLY · NO SCRIPTED CONTACT TORQUE · NO TEACHER",
        (
            f"MIN PELVIS {result['shooter_min_pelvis_height_m']:.3f} m · "
            f"TAIL WOBBLE {result['shooter_tail_wobble_index']:.4f} · NO FALL"
        ),
        "EXACT REPLAY PASS · CAUSAL GOAL DELTA · CPU MUJOCO · SIM ONLY",
    )
    paths = tuple(root / f"label-{index}.txt" for index in range(len(values)))
    for path, value in zip(paths, values, strict=True):
        path.write_text(value, encoding="utf-8")
    return paths


def _ffmpeg_command(
    *,
    ffmpeg: str,
    output: Path,
    fps: int,
    width: int,
    height: int,
    labels: tuple[Path, ...],
    clips: tuple[ThreePlayerVideoClip, ...],
) -> list[str]:
    font = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
    font_option = f"fontfile={escape_filtergraph_option(str(font))}:" if font.is_file() else ""
    scale = height / 720.0
    left = round(30 * scale)
    filters = [
        f"drawbox=x=0:y=0:w=iw:h={round(118 * scale)}:color=0x030711@0.84:t=fill",
        f"drawbox=x=0:y=h-{round(64 * scale)}:w=iw:h={round(64 * scale)}:"
        "color=0x030711@0.84:t=fill",
        f"drawtext={font_option}text=ROSClaw Soccer · NEURAL MUSCLE MEMORY:"
        f"expansion=none:x={left}:y={round(13 * scale)}:fontsize={round(32 * scale)}:"
        "fontcolor=white",
        f"drawtext={font_option}text=PASSING CANARY · NOT PROMOTED · CPU MUJOCO · SIM ONLY:"
        f"expansion=none:x={left}:y=h-{round(42 * scale)}:"
        f"fontsize={round(19 * scale)}:fontcolor=0x8DD8FF",
    ]
    offset = 0.0
    for label, clip in zip(labels, clips, strict=True):
        end = offset + clip.duration_sec
        filters.append(
            f"drawtext={font_option}textfile={escape_filtergraph_option(str(label))}:"
            f"expansion=none:x={left}:y={round(62 * scale)}:fontsize={round(20 * scale)}:"
            f"fontcolor=0x65F59A:enable='between(t,{offset:.6f},{end:.6f})'"
        )
        offset = end
    return [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pixel_format",
        "rgb24",
        "-video_size",
        f"{width}x{height}",
        "-framerate",
        str(fps),
        "-i",
        "pipe:0",
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


__all__ = [
    "render_neural_contact_canary_video",
    "validate_neural_contact_canary_video",
]
