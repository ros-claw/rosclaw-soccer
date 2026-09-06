"""Render the evidence-bound S208 pass, strike, and goalkeeper-save showcase."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, BinaryIO, cast

import numpy as np

from rosclaw_soccer.media.continuous_competitive_match_video import (
    _Clip,
    _Frame,
    _write_frames,
    _write_labels,
)
from rosclaw_soccer.media.independent_team_video import _color_player, _probe
from rosclaw_soccer.media.trajectory_render import escape_filtergraph_option
from rosclaw_soccer.providers.g1.asset_qualification import trajectory_digest
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.training.continuous_competitive_match_growth import (
    build_continuous_competitive_fixture,
)
from rosclaw_soccer.training.phase_conditioned_strike_growth import (
    validate_phase_conditioned_strike_growth,
)
from rosclaw_soccer.world.field import G1TrainingGoalSpec
from rosclaw_soccer.world.multi_player import build_g1_multi_player_stadium_model

_CLAIM = "PHYSICAL_PHASE_STRIKE_AND_GLOVE_SAVE_REPLAY"
_REQUIRED_PHASES = (1, 2, 3, 4, 5, 6)


def render_phase_conditioned_strike_video(
    *,
    exam_path: Path,
    asset_root: Path,
    output_path: Path,
    fps: int = 30,
    width: int = 1920,
    height: int = 1080,
) -> dict[str, Any]:
    """Render only a validated PASS report; video pixels never score it."""

    output = output_path.expanduser().resolve()
    manifest_path = output.with_suffix(".json")
    if (
        output.exists()
        or manifest_path.exists()
        or output.suffix.lower() != ".mp4"
        or not 20 <= fps <= 60
        or not 1280 <= width <= 3840
        or not 720 <= height <= 2160
    ):
        raise ValueError("phase-conditioned strike video output contract is invalid")
    report_path = exam_path.expanduser().resolve()
    report = validate_phase_conditioned_strike_growth(report_path)
    assessment = cast(dict[str, Any], report.get("primary_assessment"))
    gates = cast(dict[str, bool], assessment.get("gates"))
    if (
        report.get("status") != "PASS_PHASE_CONDITIONED_STRIKE_SAVE"
        or report.get("passed") is not True
        or report.get("exact_replay") is not True
        or assessment.get("passed") is not True
        or tuple(assessment.get("phase_sequence", ())) != _REQUIRED_PHASES
        or not gates
        or not all(gates.values())
    ):
        raise ValueError("phase-conditioned strike video requires the qualified PASS rung")
    artifact = cast(dict[str, str], report.get("primary_artifact"))
    trajectory_path = report_path.parent / str(artifact.get("file"))
    if not trajectory_path.is_file() or hash_bytes(trajectory_path.read_bytes()) != artifact.get(
        "file_hash"
    ):
        raise ValueError("phase-conditioned strike video trajectory changed")
    with np.load(trajectory_path, allow_pickle=False) as archive:
        trajectory = {name: np.asarray(archive[name]) for name in archive.files}
    digest = trajectory_digest(trajectory)
    if digest != artifact.get("trajectory_digest") or digest != assessment.get("trajectory_hash"):
        raise ValueError("phase-conditioned strike trajectory identity changed")
    clips = _timeline(trajectory, assessment=assessment, fps=fps)
    source_files = {
        str(report_path): hash_bytes(report_path.read_bytes()),
        str(trajectory_path): hash_bytes(trajectory_path.read_bytes()),
    }
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if ffmpeg is None or ffprobe is None:
        raise RuntimeError("ffmpeg and ffprobe are required for S208 video")
    output.parent.mkdir(parents=True, exist_ok=True)
    previous_gl = os.environ.get("MUJOCO_GL")
    os.environ["MUJOCO_GL"] = "egl"
    try:
        fixture = build_continuous_competitive_fixture(asset_root)
        goal = G1TrainingGoalSpec(**cast(dict[str, Any], report["goal"]))
        import mujoco

        model = build_g1_multi_player_stadium_model(asset_root, players=fixture.players, spec=goal)
        for player in fixture.players:
            _color_player(
                model,
                root_body_name=player.body_prefix + "pelvis",
                rgba=(0.88, 0.05, 0.04, 1.0)
                if player.agent_id.startswith("red.")
                else (0.02, 0.20, 0.92, 1.0),
            )
        model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), width)
        model.vis.global_.offheight = max(int(model.vis.global_.offheight), height)
        data = mujoco.MjData(model)
        renderer = mujoco.Renderer(model, height=height, width=width)
        try:
            with tempfile.TemporaryDirectory(prefix="rosclaw-s208-video-") as temp:
                labels = _write_labels(Path(temp), clips)
                process = subprocess.Popen(
                    _ffmpeg_command(
                        ffmpeg=ffmpeg,
                        output=output,
                        width=width,
                        height=height,
                        fps=fps,
                        clips=clips,
                        labels=labels,
                    ),
                    stdin=subprocess.PIPE,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                )
                if process.stdin is None:
                    raise RuntimeError("S208 raw video pipe is unavailable")
                try:
                    _write_frames(
                        mujoco=mujoco,
                        model=model,
                        data=data,
                        renderer=renderer,
                        players=fixture.players,
                        trajectory=trajectory,
                        clips=clips,
                        stream=cast(BinaryIO, process.stdin),
                    )
                except BrokenPipeError as error:
                    process.stdin.close()
                    process.wait()
                    stderr = (
                        process.stderr.read().decode(errors="replace") if process.stderr else ""
                    )
                    raise RuntimeError(f"S208 ffmpeg closed its input: {stderr[-3000:]}") from error
                except BaseException:
                    process.stdin.close()
                    process.kill()
                    process.wait()
                    raise
                process.stdin.close()
                stderr = process.stderr.read().decode(errors="replace") if process.stderr else ""
                if process.wait():
                    raise RuntimeError(f"S208 ffmpeg failed: {stderr[-3000:]}")
        finally:
            renderer.close()
    finally:
        if previous_gl is None:
            os.environ.pop("MUJOCO_GL", None)
        else:
            os.environ["MUJOCO_GL"] = previous_gl
    frame_count = sum(len(clip.frames) for clip in clips)
    probe = _probe(ffprobe, output)
    if (
        probe["width"] != width
        or probe["height"] != height
        or probe["fps"] != fps
        or abs(probe["frame_count"] - frame_count) > 1
    ):
        raise RuntimeError("S208 encoded video contract changed")
    manifest: dict[str, Any] = {
        "schema_version": "rosclaw_soccer.phase_conditioned_strike_video.v1",
        "claim": _CLAIM,
        "video_path": str(output),
        "video_hash": hash_bytes(output.read_bytes()),
        "source_report_path": str(report_path),
        "source_trajectory_path": str(trajectory_path),
        "source_files": source_files,
        "source_report_hash": report["report_hash"],
        "source_trajectory_digest": digest,
        "phase_sequence": list(_REQUIRED_PHASES),
        "strike_time_sec": assessment["events"]["strike_time_sec"],
        "save_time_sec": assessment["events"]["save_time_sec"],
        "peak_shot_speed_mps": assessment["metrics"]["peak_shot_speed_mps"],
        "projected_goal_y_m": assessment["metrics"]["projected_goal_y_m"],
        "projected_goal_z_m": assessment["metrics"]["projected_goal_z_m"],
        "strict_replay": True,
        "world_safe": True,
        "whole_body_g1_count": 6,
        "fps": fps,
        "width": width,
        "height": height,
        "frame_count": frame_count,
        "duration_sec": frame_count / fps,
        "camera_views": sorted({frame.camera for clip in clips for frame in clip.frames}),
        "ball_trail_is_visualization": True,
        "visualization_only": True,
        "pixels_used_for_scoring": False,
        "goal_claimed": False,
        "glove_save_claimed": True,
        "activation_ceiling": "SIM_ONLY",
        "hardware_command_sent": False,
        "commercial_use_allowed": False,
        "renderer_hash": hash_bytes(Path(__file__).read_bytes()),
    }
    manifest["manifest_hash"] = hash_json(manifest)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    validate_phase_conditioned_strike_video_manifest(manifest_path)
    return manifest


def validate_phase_conditioned_strike_video_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("S208 video manifest must be an object")
    declared = payload.pop("manifest_hash", None)
    try:
        video = Path(str(payload.get("video_path"))).expanduser().resolve()
        report_path = Path(str(payload.get("source_report_path"))).expanduser().resolve()
        trajectory_path = Path(str(payload.get("source_trajectory_path"))).expanduser().resolve()
        sources = payload.get("source_files")
        report = validate_phase_conditioned_strike_growth(report_path)
        assessment = cast(dict[str, Any], report.get("primary_assessment"))
        artifact = cast(dict[str, Any], report.get("primary_artifact"))
        if (
            declared != hash_json(payload)
            or payload.get("schema_version") != "rosclaw_soccer.phase_conditioned_strike_video.v1"
            or payload.get("claim") != _CLAIM
            or not video.is_file()
            or hash_bytes(video.read_bytes()) != payload.get("video_hash")
            or trajectory_path != report_path.parent / str(artifact.get("file"))
            or not isinstance(sources, dict)
            or sources
            != {
                str(report_path): hash_bytes(report_path.read_bytes()),
                str(trajectory_path): hash_bytes(trajectory_path.read_bytes()),
            }
            or any(
                not Path(source).is_file() or hash_bytes(Path(source).read_bytes()) != expected
                for source, expected in sources.items()
            )
            or payload.get("source_report_hash") != report.get("report_hash")
            or payload.get("source_trajectory_digest") != artifact.get("trajectory_digest")
            or payload.get("phase_sequence") != list(_REQUIRED_PHASES)
            or payload.get("phase_sequence") != assessment.get("phase_sequence")
            or payload.get("strike_time_sec") != assessment.get("events", {}).get("strike_time_sec")
            or payload.get("save_time_sec") != assessment.get("events", {}).get("save_time_sec")
            or payload.get("peak_shot_speed_mps")
            != assessment.get("metrics", {}).get("peak_shot_speed_mps")
            or payload.get("projected_goal_y_m")
            != assessment.get("metrics", {}).get("projected_goal_y_m")
            or payload.get("projected_goal_z_m")
            != assessment.get("metrics", {}).get("projected_goal_z_m")
            or payload.get("strict_replay") is not True
            or payload.get("world_safe") is not True
            or payload.get("whole_body_g1_count") != 6
            or payload.get("ball_trail_is_visualization") is not True
            or payload.get("visualization_only") is not True
            or payload.get("pixels_used_for_scoring") is not False
            or payload.get("goal_claimed") is not False
            or payload.get("glove_save_claimed") is not True
            or payload.get("activation_ceiling") != "SIM_ONLY"
            or payload.get("hardware_command_sent") is not False
            or payload.get("commercial_use_allowed") is not False
            or payload.get("renderer_hash") != hash_bytes(Path(__file__).read_bytes())
        ):
            raise ValueError("S208 video integrity or authority contract is invalid")
    finally:
        payload["manifest_hash"] = declared
    return cast(dict[str, Any], payload)


def _timeline(
    trajectory: dict[str, np.ndarray], *, assessment: dict[str, Any], fps: int
) -> tuple[_Clip, ...]:
    start = float(trajectory["time"][0])
    end = float(trajectory["time"][-1])
    strike = float(assessment["events"]["strike_time_sec"])
    save = float(assessment["events"]["save_time_sec"])

    def hold(timestamp: float, duration_sec: float, camera: str) -> tuple[_Frame, ...]:
        return tuple(_Frame(timestamp, camera) for _ in range(round(duration_sec * fps)))

    def play(first: float, last: float, speed: float, camera: str) -> tuple[_Frame, ...]:
        duration = max(0.0, last - first) / speed
        return tuple(
            _Frame(min(last, first + index / fps * speed), camera)
            for index in range(max(1, math.ceil(duration * fps)))
        )

    return (
        _Clip("S208 · PHASE-CONDITIONED PHYSICAL STRIKE", hold(start, 1.0, "wide")),
        _Clip(
            "ONE CLOCK · SIX G1 · PASS + RUN-ONTO RECEIVE",
            play(start, min(3.35, end), 1.0, "broadcast"),
        ),
        _Clip(
            "CAPTURE → ORIENT → PLANT · NO BALL OR ROOT TELEPORT",
            play(2.80, min(strike + 0.10, end), 0.72, "touchline"),
        ),
        _Clip(
            "7.04 m/s FOOT STRIKE → ON-TARGET GLOVE SAVE · 0.35x",
            play(max(start, strike - 0.25), min(end, save + 0.38), 0.35, "counter"),
        ),
        _Clip(
            "RECOVER → COMPLETE · SHOOTER REMAINS UPRIGHT",
            play(max(start, save - 0.10), end, 0.65, "broadcast"),
        ),
        _Clip(
            "STRICT REPLAY PASS · ALL PHYSICS GATES PASS · SIM_ONLY",
            hold(end, 1.5, "wide"),
        ),
    )


def _ffmpeg_command(
    *,
    ffmpeg: str,
    output: Path,
    width: int,
    height: int,
    fps: int,
    clips: tuple[_Clip, ...],
    labels: tuple[Path, ...],
) -> list[str]:
    scale = height / 720.0
    left = round(28 * scale)
    filters = [
        f"drawbox=x=0:y=0:w=iw:h={round(124 * scale)}:color=0x020714@0.82:t=fill",
        f"drawbox=x=0:y=h-{round(72 * scale)}:w=iw:h={round(72 * scale)}:"
        "color=0x020714@0.84:t=fill",
        f"drawtext=font='DejaVu Sans':text='ROSClaw Soccer · S208 PHYSICAL STRIKE CHAIN':"
        f"expansion=none:x={left}:y={round(12 * scale)}:fontsize={round(31 * scale)}:"
        "fontcolor=white",
        f"drawtext=font='DejaVu Sans':text='CPU MUJOCO · STRICT REPLAY PASS · WORLD SAFE PASS':"
        f"expansion=none:x={left}:y={round(50 * scale)}:fontsize={round(18 * scale)}:"
        "fontcolor=0xFFC857",
        "drawtext=font='DejaVu Sans':text='PASS  |  RECEIVE  |  PHASE STRIKE  |  "
        "GLOVE SAVE  |  RECOVERY':"
        f"expansion=none:x={left}:y=h-{round(47 * scale)}:fontsize={round(18 * scale)}:"
        "fontcolor=0x8ED9FF",
    ]
    offset = 0.0
    for clip, label in zip(clips, labels, strict=True):
        end = offset + len(clip.frames) / fps
        filters.append(
            f"drawtext=font='DejaVu Sans':textfile={escape_filtergraph_option(str(label))}:"
            f"expansion=none:x={left}:y={round(84 * scale)}:fontsize={round(19 * scale)}:"
            f"fontcolor=0x68F596:enable='between(t,{offset:.6f},{end:.6f})'"
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


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exam", required=True, type=Path)
    parser.add_argument("--asset-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    return parser


def main() -> None:
    arguments = _parser().parse_args()
    value = render_phase_conditioned_strike_video(
        exam_path=arguments.exam,
        asset_root=arguments.asset_root,
        output_path=arguments.output,
        fps=arguments.fps,
        width=arguments.width,
        height=arguments.height,
    )
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = [
    "render_phase_conditioned_strike_video",
    "validate_phase_conditioned_strike_video_manifest",
]
